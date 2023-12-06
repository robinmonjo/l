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

  defn normalize(t) do
    (t - Nx.mean(t)) / Nx.standard_deviation(t)
  end

  def size(container) do
    container_byte_size(container)
    |> human_readable_size()
  end

  defp container_byte_size(%Nx.Tensor{} = t) do
    Nx.byte_size(t)
  end

  defp container_byte_size(%{} = map) do
    Enum.reduce(map, 0, fn
      {_, %Nx.Tensor{} = t}, acc -> acc + Nx.byte_size(t)
      {_, %{} = m}, acc -> acc + container_byte_size(m)
    end)
  end

  defp human_readable_size(byte_size, base \\ 1024) do
    i = if byte_size == 0, do: 0, else: floor(:math.log(byte_size) / :math.log(base))
    size = byte_size / :math.pow(base, i)
    "#{Float.round(size, 2)} #{Enum.at(["B", "kB", "MB", "GB", "TB"], i)}"
  end
end
