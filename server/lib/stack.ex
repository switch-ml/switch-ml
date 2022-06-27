defmodule Switchml.Stack do
  def start_link do
    # __MODULE__ here would be HelloBlockchain.Monitor

    Agent.start_link(fn -> %{} end, name: __MODULE__)
  end

  def put(key, value) do
    Agent.update(__MODULE__, &Map.put_new(&1, key, value))
  end

  def get(key) do
    Agent.get(__MODULE__, &Map.get(&1, key, []))
  end
end
