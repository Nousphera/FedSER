import json
import time
from dataclasses import asdict
import numpy as np
import pytest
pytest.importorskip("flwr")
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, Metadata, RecordDict
from fedser.client_app import train
from fedser.train import RunConfig, make_data
from fedser.model import build_model


def test_flower_message_fit_persists_participation_and_reports_exact_weight():
    config = RunConfig(clients=2, rounds=2, synthetic_per_client=8, labeled_fraction=.5, batch_size=4)
    _, _, shape = make_data(config)
    arrays = ArrayRecord(build_model(shape).get_weights())
    context = Context(1, 17, {"partition-id": 0, "num-partitions": 2}, RecordDict(), {})
    for round_number in (1, 2):
        msg = Message(content=RecordDict({"arrays": arrays, "config": ConfigRecord({
            "experiment": json.dumps(asdict(config)), "server-round": round_number})}),
            metadata=Metadata(run_id=1, message_id=str(round_number), src_node_id=0,
                              dst_node_id=17, reply_to_message_id="", group_id="",
                              created_at=time.time(), ttl=3600., message_type="train"))
        response = train(msg, context)
        assert response.content["metrics"]["num-examples"] == 8
        assert context.state["participation"]["completed"] == round_number
        assert response.content["metrics"]["steps"] == 1
        arrays = response.content["arrays"]
    assert response.content["metrics"]["threshold"] > .5
