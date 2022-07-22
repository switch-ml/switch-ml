defmodule SwitchMl.Endpoint do
  use GRPC.Endpoint

  intercept(GRPC.Logger.Server)
  run(SwitchMl.Server)
end
