defmodule L.Layer do
  alias L.ComputeStats

  defstruct [
    :name,
    :kernels,
    :stacked,
    :mean,
    :std,
    :hist
  ]

  def append(layer, kernel) do
    kernels = [kernel | layer.kernels || []]
    %{layer | kernels: kernels}
  end

  def data(layer, stat) do
    Map.get(layer, stat)
    |> Nx.to_list()
    |> Enum.reverse() # stats are appended
    |> Enum.with_index()
    |> Enum.map(fn {value, i} ->
      %{
        x: i,
        y: value,
        layer: layer.name
      }
    end)
  end

  def compute(layer, stat) do
    if Map.get(layer, stat) do
      layer
    else
      layer = ensure_concat(layer)
      %{layer | stat => apply(ComputeStats, stat, [layer.stacked])}
    end
  end

  def stacked_histograms(layer) do
    layer
    |> compute(:hist)
    |> Map.get(:hist)
    |> Nx.transpose()
    |> Nx.log1p()
  end

  defp ensure_concat(layer) do
    case layer do
      %{stacked: nil} ->
        %{layer | stacked: Nx.stack(layer.kernels)}
      _ -> layer
    end
  end
end
