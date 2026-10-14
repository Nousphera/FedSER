"""Flower Message API client with durable per-client participation state."""
import json
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .train import RunConfig, local_train, make_data

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = RunConfig(**json.loads(msg.content["config"]["experiment"]))
    cid = int(context.node_config["partition-id"])
    if int(context.node_config["num-partitions"]) != config.clients:
        raise ValueError("Flower node count must equal experiment client count")
    completed = int(msg.content["config"]["server-round"]) - 1
    state = context.state.get("participation", ConfigRecord({"completed": 0}))
    client_completed = int(state["completed"])
    clients, _, shape = make_data(config)
    weights, count, metrics = local_train(msg.content["arrays"].to_numpy_ndarrays(), clients[cid],
                                          shape, config, completed, client_completed, cid)
    context.state["participation"] = ConfigRecord({"completed": client_completed + 1})
    content = RecordDict({"arrays": ArrayRecord(weights),
                          "metrics": MetricRecord({"num-examples": count, "client-id": cid, **metrics})})
    return Message(content=content, reply_to=msg)
