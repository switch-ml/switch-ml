defmodule Switchml.SendWeightsRequest do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:fit_res, :round, :client_id]

  field(:fit_res, 2, type: Switchml.FitRes, json_name: "fitRes")
  field(:round, 3, type: :int64)
  field(:client_id, 4, type: :string, json_name: "clientId")
end

defmodule Switchml.SendWeightsResponse do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:parameters]

  field(:parameters, 1, type: Switchml.Parameters)
end

defmodule Switchml.FetchWeightsRequest do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct []
end

defmodule Switchml.FetchWeightsResponse do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:parameters]

  field(:parameters, 1, type: Switchml.Parameters)
end

defmodule Switchml.Parameters do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:tensors, :tensor_type]

  field(:tensors, 1, repeated: true, type: :bytes)
  field(:tensor_type, 2, type: :string, json_name: "tensorType")
end

defmodule Switchml.EvaluateRes.MetricsEntry do
  @moduledoc false
  use Protobuf, map: true, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:key, :value]

  field(:key, 1, type: :string)
  field(:value, 2, type: :float)
end

defmodule Switchml.EvaluateRes do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:loss, :num_examples, :metrics]

  field(:loss, 2, type: :float)
  field(:num_examples, 3, type: :int64, json_name: "numExamples")
  field(:metrics, 4, repeated: true, type: Switchml.EvaluateRes.MetricsEntry, map: true)
end

defmodule Switchml.FitRes.MetricsEntry do
  @moduledoc false
  use Protobuf, map: true, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:key, :value]

  field(:key, 1, type: :string)
  field(:value, 2, type: :float)
end

defmodule Switchml.FitRes do
  @moduledoc false
  use Protobuf, protoc_gen_elixir_version: "0.10.0", syntax: :proto3

  defstruct [:parameters, :num_examples, :metrics]

  field(:parameters, 2, type: Switchml.Parameters)
  field(:num_examples, 3, type: :int64, json_name: "numExamples")
  field(:metrics, 4, repeated: true, type: Switchml.FitRes.MetricsEntry, map: true)
end

defmodule Switchml.SwitchmlWeightsService.Service do
  @moduledoc false
  use GRPC.Service, name: "switchml.SwitchmlWeightsService", protoc_gen_elixir_version: "0.10.0"

  rpc(:SendWeights, Switchml.SendWeightsRequest, Switchml.SendWeightsResponse)

  rpc(:FetchWeights, Switchml.FetchWeightsRequest, Switchml.FetchWeightsResponse)
end

defmodule Switchml.SwitchmlWeightsService.Stub do
  @moduledoc false
  use GRPC.Stub, service: Switchml.SwitchmlWeightsService.Service
end
