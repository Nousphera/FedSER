"""Native Flower client with persistent participation and centralized Adam state."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .config import defaults, validate
from .data import load_data, partition_records
from .models import build_model
from .training import seed_everything, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = context.node_config["partition-id"]
    if type(cid) != int or cid < 0:
        raise ValueError("Invalid client partition")
    incoming = msg.content["config"]
    rnd, phase = incoming["round"], incoming["phase"]
    if type(rnd) != int or not 0 <= rnd <= cfg["num-rounds"] or phase not in {"identify", "train"}:
        raise ValueError("Invalid experiment message")
    if (phase == "identify") != (rnd == 0):
        raise ValueError("Invalid initialization round")
    if phase == "identify":
        return Message(RecordDict({"metrics": MetricRecord({"client-id": cid})}), reply_to=msg)
    progress = context.state.get("progress", ConfigRecord({"completed": 0, "last-round": 0}))
    completed = incoming["client-completed"]
    if (type(completed) != int or completed != progress["completed"]
            or not 0 <= completed < rnd or rnd <= progress["last-round"]):
        raise ValueError("Client participation state does not match the server round")
    seed_everything(cfg["seed"] + cid, cfg["torch-threads"])
    data = load_data(cfg["data-root"], cfg)
    partitions = partition_records(data.train, cfg)
    if cid >= len(partitions):
        raise ValueError("Invalid client partition")
    records = partitions[cid]
    model = build_model()
    state = msg.content["model"].to_torch_state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Received nonfinite model")
    model.load_state_dict(state, strict=True)
    optimizer = context.state["optimizer"].to_torch_state_dict() if "optimizer" in context.state else None
    metrics, optimizer = train_local(model, records, cfg, cfg["seed"] + rnd * len(partitions) + cid,
                                     rnd, optimizer, completed)
    if cfg["method"] == "centralized":
        context.state["optimizer"] = ArrayRecord(optimizer)
    context.state["progress"] = ConfigRecord({"completed": completed + 1, "last-round": rnd})
    return Message(RecordDict({
        "model": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"client-id": cid, "num-examples": len(records),
                                 "client-completed": completed + 1, **metrics}),
    }), reply_to=msg)
