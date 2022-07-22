defmodule SwitchMl.Server do
  use GRPC.Server, service: Switchml.SwitchmlService.Service

  def fetch_weights(_request, _stream) do
    opts = [
      interceptors: [GRPC.Logger.Client],
      adapter_opts: %{http2_opts: %{keepalive: :infinity}}
    ]

    {:ok, channel} = GRPC.Stub.connect("localhost:8000", opts)
    req = Switchml.FetchWeightsRequest.new()
    {:ok, response} = channel |> Switchml.SwitchmlService.Stub.fetch_weights(req)
    Switchml.FetchWeightsResponse.new(response)
  end

  def send_weights(request, stream) do
    opts = [
      interceptors: [GRPC.Logger.Client],
      adapter_opts: %{http2_opts: %{keepalive: :infinity}}
    ]

    {:ok, channel} = GRPC.Stub.connect("localhost:8000", opts)
    {:ok, response} = channel |> Switchml.SwitchmlService.Stub.send_weights(request)

    Enum.each(response, fn res ->
      case res do
        {:ok, result} -> GRPC.Server.send_reply(stream, result)
        {:error} -> "Uh oh!"
        _ -> "Catch all"
      end
    end)
  end
end
