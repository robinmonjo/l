defmodule NxExtra do
  import Nx.Defn

  defn histogram(tensor, opts \\ []) do
    opts = keyword!(opts, [:bins, :min, :max])

    t = Nx.flatten(tensor)

    bins = opts[:bins]
    min = opts[:min]
    max = opts[:max]

    {bins, min, max}

    bin_width =
      Nx.subtract(max, min)
      |> Nx.add(1)
      |> Nx.divide(bins)

    bin_indices =
      t
      |> Nx.subtract(min)
      |> Nx.divide(bin_width)
      |> Nx.floor()
      |> Nx.as_type(:s32)
      |> Nx.new_axis(1)

    {size} = Nx.shape(t)

    zeros_dim = Nx.broadcast(Nx.tensor(0), {size, 1})
    indices = Nx.concatenate([zeros_dim, bin_indices], axis: 1)

    histogram = Nx.broadcast(Nx.tensor(0), {1, bins})

    update = Nx.broadcast(Nx.tensor(1), {size})

    Nx.indexed_add(histogram, indices, update)
    |> Nx.flatten()
  end
end
