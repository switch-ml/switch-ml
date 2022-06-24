from flask import Flask, Response
import grpc


app = Flask(__name__)


class SwitchMlCient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = "localhost"
        self.server_port = 8000

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            "{}:{}".format(self.host, self.server_port)
        )


app.config["client"] = SwitchMlCient()


@app.route("/")
def users_get():
    print(app.config["client"].__dict__)
    return Response("SwitchML Client", content_type="application/json")
