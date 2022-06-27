defmodule Switchml.MixProject do
  use Mix.Project

  def project do
    [
      app: :switchml,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      mod: {Switchml, []},
      extra_applications: [:logger, :grpc]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:grpc, "~> 0.3.1"},
      {:protobuf, "~> 0.10.0"},
      {:google_protos, "~> 0.2.0"},
      {:cowlib, "~> 2.8", hex: :grpc_cowlib, override: true}
    ]
  end
end
