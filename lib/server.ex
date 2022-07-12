defmodule Switchml.Server do
  use GRPC.Server, service: Switchml.SwitchmlWeightsService.Service

  def send_weights(request, _stream) do
    IO.inspect("RECEIVED WEIGHTS...")

    # , interceptors: [GRPC.Logger.Server]

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    req =
      Switchml.SendWeightsRequest.new(
        fit_res: request.fit_res,
        round: request.round
      )

    {:ok, response} = channel |> Switchml.SwitchmlWeightsService.Stub.send_weights(req)

    Switchml.SendWeightsResponse.new()
  end

  def fetch_weights(request, _stream) do
    IO.inspect("FETCHING WEIGHTS...")

    {:ok, channel} = GRPC.Stub.connect("localhost:8000")

    req = Switchml.FetchWeightsRequest.new(request: request.request)

    {:ok, response} = channel |> Switchml.SwitchmlWeightsService.Stub.fetch_weights(req)

    IO.inspect("RECEIVED WEIGHTS AND SENDING TO CLIENT")

    Switchml.FetchWeightsResponse.new(parameters: response.parameters, status: response.status)
  end
end
