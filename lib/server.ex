defmodule Switchml.ModelService.Server do
  use GRPC.Server, service: Switchml.SwitchmlWeightsService.Service

  def send_weights(request, _stream) do
    IO.inspect("RECEIVED WEIGHTS...")

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    request = Switchml.SendWeightsRequest.new(weights: request.weights)

    {:ok, response} = channel |> Switchml.SwitchmlWeightsService.Stub.send_weights(request)

    Switchml.SendWeightsResponse.new()
  end

  def fetch_weights(request, _stream) do
    IO.inspect("FETCHING WEIGHTS...")

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    request = Switchml.FetchWeightsRequest.new()

    {:ok, response} = channel |> Switchml.SwitchmlWeightsService.Stub.fetch_weights(request)

    Switchml.FetchWeightsResponse.new(weights: response.weights)
  end
end
