defmodule AxonExtra.Activations do
  import Nx.Defn

  defn relu(t, opts \\ []) do
    opts = keyword!(opts, leak: 0.0, sub: 0.0)

    leak = opts[:leak]
    sub = opts[:sub]

    relu =
      if leak == 0 do
        Axon.Activations.relu(t)
      else
        Axon.Activations.leaky_relu(t, alpha: leak)
      end

    if sub == 0, do: relu, else: Nx.subtract(relu, sub)
  end
end
