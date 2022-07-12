defmodule Switchml.Server do
  use GRPC.Server, service: Switchml.SwitchmlWeightsService.Service

  def send_weights(request, _stream) do
    IO.inspect("RECEIVED WEIGHTS...")

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    req =
      Switchml.SendWeightsRequest.new(
        fit_res: request.fit_res,
        round: request.round,
        client_id: request.client_id
      )

    {:ok, response} = channel |> Switchml.SwitchmlWeightsService.Stub.send_weights(req)

    Switchml.SendWeightsResponse.new(parameters: response.parameters)
  end

  def fetch_weights(request, _stream) do
    IO.inspect("FETCHING WEIGHTS...")

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    req = Switchml.FetchWeightsRequest.new()

    {:ok, response} = channel |> Switchml.SwitchmlWeightsService.Stub.fetch_weights(req)

    IO.inspect("RECEIVED WEIGHTS AND SENDING TO CLIENT")

    Switchml.FetchWeightsResponse.new(parameters: response.parameters)
  end
end
