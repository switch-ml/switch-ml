defmodule Switchml.SendWeightsRequest do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field :weights, 1, type: :string
end
defmodule Switchml.SendWeightsResponse do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3
end
defmodule Switchml.FetchWeightsRequest do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3
end
defmodule Switchml.FetchWeightsResponse do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field :weights, 1, type: :string
end
defmodule Switchml.WeightService.Service do
  @moduledoc false
  use GRPC.Service, name: "switchml.WeightService", protoc_gen_elixir_version: "0.10.0"

  rpc :SendWeights, Switchml.SendWeightsRequest, Switchml.SendWeightsResponse

  rpc :FetchWeights, Switchml.FetchWeightsRequest, Switchml.FetchWeightsResponse
end

defmodule Switchml.WeightService.Stub do
  @moduledoc false
  use GRPC.Stub, service: Switchml.WeightService.Service
end
