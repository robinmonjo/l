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

  def losses_for_lrs() do
    State.get(:losses_for_lrs)
    |> line_chart()
    |> Vl.encode_field(:x, "lr", type: :quantitative, scale: [type: :log])
    |> Vl.encode_field(:y, "loss", type: :quantitative)
    |> Kino.VegaLite.new()
  end

  def lrs() do
    State.get(:lrs)
    |> line_chart()
    |> Vl.encode_field(:x, "iteration", type: :quantitative)
    |> Vl.encode_field(:y, "lr", type: :quantitative)
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

  defp layers() do
    State.get(:monitored_layers)
    # layers in order
    |> Enum.reverse()
  end

  def hists() do
    hists = for l <- layers(), do: hist(l)
    Kino.Layout.grid(hists, columns: 3)
  end

  def hist(layer, img_scale \\ 5) do
    hists_tensor = State.stacked_histograms(layer)

    {w, h} = Nx.shape(hists_tensor)
    {w, h} = {img_scale * w, img_scale * h}

    max = Nx.reduce_max(hists_tensor)

    # 0 to 255 grayscale
    # using 245 so we have no white pixels and better see
    # the full size of the image
    scale = Nx.divide(245, max)

    img =
      hists_tensor
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

  def deads() do
    charts = for l <- layers(), do: dead(l)
    Kino.Layout.grid(charts, columns: 3)
  end

  def dead(layer) do
    hists_tensor = State.stacked_histograms(layer)

    points =
      hists_tensor[0]
      |> Nx.divide(Nx.sum(hists_tensor, axes: [0]))
      |> Nx.to_list()
      |> Enum.with_index()
      |> Enum.map(fn {value, i} -> %{iteration: i, value: value} end)

    chart =
      Vl.new(width: 200, height: 66)
      |> Vl.mark(:line)
      |> Vl.data_from_values(points)
      |> Vl.encode_field(:x, "iteration", type: :quantitative)
      |> Vl.encode_field(:y, "value", type: :quantitative)
      |> Kino.VegaLite.new()

    Kino.Layout.grid([Kino.Text.new(layer), chart], columns: 1)
  end
end
