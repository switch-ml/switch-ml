defmodule Switchml.Server do
  use GRPC.Server, service: Switchml.WeightService.Service

  def send_weights(request, _stream) do
    IO.inspect(request.weights, "RECEIEVED WEIGHTS FROM CLIENT")

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    request = Switchml.SendWeightsRequest.new(weights: request.weights)

    response = channel |> Switchml.WeightService.Stub.send_weights(request)

    IO.inspect(response, "WEIGHTS SENT TO SERVER")

    Switchml.SendWeightsResponse.new()
  end

  def fetch_weights(_request, _stream) do
    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    request = Switchml.FetchWeightsRequest.new()

    response = channel |> Switchml.WeightService.Stub.fetch_weights(request)

    IO.inspect(response, "RECEIVED WEIGHTS FROM SERVER")

    Switchml.FetchWeightsResponse.new(weights: response)
  end
end
