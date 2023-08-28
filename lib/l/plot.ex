defmodule L.Plot do
  alias VegaLite, as: Vl
  alias L.State

  def metric(metric \\ "loss") do
    line_chart()
    |> Vl.encode_field(:x, "step", type: :quantitative)
    |> Vl.encode_field(:y, metric, type: :quantitative)
    |> Kino.VegaLite.new()
    |> Kino.render()
  end

  def lrs() do
    State.get(:lrs)
    |> line_chart()
    |> Vl.encode_field(:x, "lr", type: :quantitative, scale: [type: :log])
    |> Vl.encode_field(:y, "loss", type: :quantitative)
    |> Kino.VegaLite.new()
  end

  def stats(layer \\ nil) do
    Kino.Layout.grid(
      [
        mean(layer),
        std(layer)
      ],
      columns: 1
    )
  end

  def mean(layer \\ nil) do
    plot(:mean, layer)
  end

  def std(layer \\ nil) do
    plot(:std, layer)
  end

  defp plot(type, layer) do
    State.get(:activations_stats)
    |> Enum.filter(fn stat ->
      if layer do
        stat.type == type && stat.layer == layer
      else
        stat.type == type
      end
    end)
    |> line_chart()
    |> Vl.encode_field(:x, "iteration", type: :quantitative)
    |> Vl.encode_field(:y, "value", type: :quantitative, axis: [title: type])
    |> Vl.encode_field(:color, "layer", type: :nominal)
    |> Kino.VegaLite.new()
  end

  defp line_chart() do
    Vl.new(width: 600, height: 200)
    |> Vl.mark(:line)
  end

  defp line_chart(points) do
    line_chart()
    |> Vl.data_from_values(points)
  end

  def hists() do
    layers =
      State.get(:monitored_layers)
      # layers in order
      |> Enum.reverse()

    hists = for l <- layers, do: hist(l)
    Kino.Layout.grid(hists, columns: 3)
  end

  def hist(layer, img_scale \\ 5) do
    hist_tensor =
      State.get(:activations_stats)
      |> Enum.filter(&(&1.type == :hist && &1.layer == layer))
      # iterations 0 is last so putting it back first
      |> Enum.reverse()
      |> Enum.map(& &1.value)
      |> Nx.stack()
      |> Nx.transpose()
      |> Nx.log1p()

    {w, h} = Nx.shape(hist_tensor)
    {w, h} = {img_scale * w, img_scale * h}

    max = Nx.reduce_max(hist_tensor)

    # 0 to 255 grayscale
    # using 245 so we have no white pixels and better see
    # the full size of the image
    scale = Nx.divide(245, max)

    img =
      hist_tensor
      # max becomes 0 (darker)
      |> Nx.subtract(max)
      |> Nx.abs()
      |> Nx.reverse(axes: [0])
      |> Nx.multiply(scale)
      |> Nx.as_type(:u8)
      # adding channel dimension
      |> Nx.new_axis(-1)
      # upsampling
      |> Nx.new_axis(0)
      |> Axon.Layers.resize(size: {w, h}, channels: :last, method: :nearest)
      |> Nx.reshape({w, h, 1})
      |> Kino.Image.new()

    Kino.Layout.grid([Kino.Text.new(layer), img], columns: 1)
  end
end
