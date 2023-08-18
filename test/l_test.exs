defmodule LTest do
  use ExUnit.Case
  doctest L

  test "greets the world" do
    assert L.hello() == :world
  end
end
