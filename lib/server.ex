defmodule Switchml.ModelService.Server do
  use GRPC.Server, service: Switchml.SwitchmlWeightsService.Service

  def send_weights(request, _stream) do
    Switchml.SendWeightsResponse.new()
  end

  def fetch_weights(request, _stream) do
    Switchml.FetchWeightsResponse.new(weights: "Weights from server")
  end
end
