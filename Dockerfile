FROM elixir:latest as build

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /app

COPY . /app

WORKDIR /app

# install Hex + Rebar
RUN mix do local.hex --force, local.rebar --force

# dependencies cleaning
RUN mix deps.clean --all

# Fetching dependencies
RUN mix deps.get --force

RUN mix deps.update --all

# Compile the project
RUN mix do compile

CMD ["iex", "-S", "mix"]