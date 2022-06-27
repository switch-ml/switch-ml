# Switchml

**TODO: Add description**

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `switchml` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:switchml, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/switchml](https://hexdocs.pm/switchml).

## ProtoBuf Conf

### Python

`python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. priv/service.proto`

### Elixir

`protoc --elixir_out=plugins=grpc:. priv/service.proto`
