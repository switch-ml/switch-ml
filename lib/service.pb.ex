defmodule Switchml.SendWeightsRequest do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:fit_res, 1, type: Switchml.FitRes, json_name: "fitRes")
  field(:eval_res, 2, type: Switchml.EvalRes, json_name: "evalRes")
  field(:round, 3, type: :string)
end

defmodule Switchml.SendWeightsResponse.ConfigEntry do
  @moduledoc false
  use Protobuf, map: true, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:key, 1, type: :string)
  field(:value, 2, type: :float)
end

defmodule Switchml.SendWeightsResponse do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:parameters, 1, type: Switchml.Parameters)
  field(:config, 3, repeated: true, type: Switchml.SendWeightsResponse.ConfigEntry, map: true)
end

defmodule Switchml.FetchWeightsRequest do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3
end

defmodule Switchml.FetchWeightsResponse.ConfigEntry do
  @moduledoc false
  use Protobuf, map: true, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:key, 1, type: :string)
  field(:value, 2, type: :float)
end

defmodule Switchml.FetchWeightsResponse do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:parameters, 1, type: Switchml.Parameters)
  field(:config, 3, repeated: true, type: Switchml.FetchWeightsResponse.ConfigEntry, map: true)
end

defmodule Switchml.Parameters do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:tensors, 1, repeated: true, type: :bytes)
  field(:tensor_type, 2, type: :string, json_name: "tensorType")
end

defmodule Switchml.FitRes.MetricsEntry do
  @moduledoc false
  use Protobuf, map: true, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:key, 1, type: :string)
  field(:value, 2, type: :float)
end

defmodule Switchml.FitRes do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:parameters, 2, type: Switchml.Parameters)
  field(:num_examples, 3, type: :int64, json_name: "numExamples")
  field(:metrics, 4, repeated: true, type: Switchml.FitRes.MetricsEntry, map: true)
end

defmodule Switchml.EvalRes.MetricsEntry do
  @moduledoc false
  use Protobuf, map: true, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:key, 1, type: :string)
  field(:value, 2, type: :float)
end

defmodule Switchml.EvalRes do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  field(:loss, 1, type: :float)
  field(:num_examples, 2, type: :int64, json_name: "numExamples")
  field(:metrics, 3, repeated: true, type: Switchml.EvalRes.MetricsEntry, map: true)
end

defmodule Switchml.SwitchmlService.Service do
  @moduledoc false
  use GRPC.Service, name: "switchml.SwitchmlService", protoc_gen_elixir_version: "0.10.0"

  rpc(:SendWeights, Switchml.SendWeightsRequest, stream(Switchml.SendWeightsResponse))

  rpc(:FetchWeights, Switchml.FetchWeightsRequest, Switchml.FetchWeightsResponse)
end

defmodule Switchml.SwitchmlService.Stub do
  @moduledoc false
  use GRPC.Stub, service: Switchml.SwitchmlService.Service
end
