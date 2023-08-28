defmodule L do
  alias L.{State, Plot}
  alias Axon.Loop

  def trainer(model, loss, optimizer, opts \\ []) do
    State.init()

    opts = Keyword.validate!(opts, monitored_layers: [])
    loop_trainer(model, loss, optimizer, opts)
  end

  defp loop_trainer(model, loss, optimizer, monitored_layers: []) do
    fun = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])

    model
    |> Loop.trainer(loss, optimizer, log: 1)
    |> Loop.handle_event(:started, fn state ->
      State.loop_started
      {:continue, state}
    end)
    |> Loop.metric(fun, "iteration_loss", fn _, obs, _ -> obs end)
    |> halt_on_crazy()
  end

  defp loop_trainer(model, loss, optimizer, monitored_layers: layers) do
    mapping = nodes_id_name_mapping(model)

    Axon.map_nodes(model, fn %Axon.Node{id: id} = node ->
      name = mapping[id]

      if Enum.member?(layers, name) do
        State.add_monitored_layer(name)
        fun = &State.append_activations_stats(name, &1)
        Axon.attach_hook(node, fun, on: :forward)
      else
        node
      end
    end)
    |> loop_trainer(loss, optimizer, monitored_layers: [])
  end

  # Plot default running average loss
  def plot_loss(%Loop{} = loop) do
    loop
    |> Loop.kino_vega_lite_plot(Plot.metric(), "loss")
  end

  # Per iteration loss
  def plot_iteration_loss(%Loop{} = loop) do
    loop
    |> Loop.kino_vega_lite_plot(Plot.metric("iteration_loss"), "iteration_loss")
  end

  # Halt on crazy
  def halt_on_crazy(%Loop{} = loop) do
    loop
    |> Loop.handle_event(:iteration_completed, &halt_if_loss_is_nan/1)
  end

  defp halt_if_loss_is_nan(state) do
    if Nx.to_number(Nx.is_nan(state.step_state.loss)) == 1 do
      IO.puts("\nLoop halted by halt_on_crazy")
      {:halt_loop, state}
    else
      {:continue, state}
    end
  end

  # LR finder
  def lr_finder_mode(%Loop{} = loop, lr_mult \\ 1.3) do
    loop
    |> Loop.handle_event(:iteration_completed, fn state ->
      increase_learning_rate(state, lr_mult)
    end)
  end

  defp increase_learning_rate(%Loop.State{} = state, lr_mult) do
    loss = Nx.to_number(state.metrics["iteration_loss"])
    prev_min_loss = Map.get(state.handler_metadata, :min_loss, 10_000)

    min_loss =
      if loss < prev_min_loss do
        loss
      else
        prev_min_loss
      end

    lrs = Map.get(state.handler_metadata, :lrs, [])

    if loss > 3 * prev_min_loss do
      State.put_lrs(lrs)
      Kino.render(Plot.lrs())
      {:halt_loop, state}
    else
      # sgd optizer lr is stored this way, might be different for other
      # optimizers
      {%{scale: scale}} = state.step_state.optimizer_state

      point = %{
        loss: loss,
        lr: abs(Nx.to_number(scale))
      }

      lrs = [point | lrs]

      next_metadata =
        state.handler_metadata
        |> Map.put(:min_loss, min_loss)
        |> Map.put(:lrs, lrs)

      next_scale = Nx.multiply(scale, lr_mult)

      next_state =
        state
        |> put_in(
          [Access.key!(:step_state), Access.key!(:optimizer_state)],
          {%{scale: next_scale}}
        )
        |> Map.put(:handler_metadata, next_metadata)

      {:continue, next_state}
    end
  end

  defp nodes_id_name_mapping(model) do
    {mapping, _} =
      Axon.reduce_nodes(model, {%{}, %{}}, fn node, {mapping, counts} ->
        count = Map.get(counts, node.op, 0)
        {Map.put(mapping, node.id, "#{node.op}_#{count}"), Map.put(counts, node.op, count + 1)}
      end)

    mapping
  end

  # Storage to access after a training
  defmodule State do
    use GenServer

    defstruct [
      :loop_started,
      :activations_stats,
      :nb_monitored_layers,
      :monitored_layers,
      :current_layer,
      :iteration,
      :lrs
    ]

    def init() do
      case GenServer.start_link(__MODULE__, empty_state(), name: __MODULE__) do
        {:ok, _} -> :ok
        {:error, {:already_started, _}} ->
          # loop is already here, should we reset it ?
          case get() do
            %State{loop_started: true} ->
              # loop was started, the state in it is from previous training
              reset()
              :ok
            _ -> :ok
          end
      end
    end

    defp empty_state() do
      %State{
        loop_started: false,
        activations_stats: [],
        monitored_layers: [],
        nb_monitored_layers: 0,
        current_layer: 0,
        iteration: 0,
        lrs: []
      }
    end

    # API
    def get(), do: call(:get)
    def get(key), do: Map.get(get(), key)

    def reset(), do: call(:reset)
    def add_monitored_layer(name), do: call({:add_monitored_layer, name})
    def loop_started(), do: call(:loop_started)

    def put_lrs(lrs), do: cast({:put_lrs, lrs})
    def append_activations_stats(layer_name, kernel) do
      cast({:append_activations_stats, layer_name, kernel})
    end

    defp call(args), do: GenServer.call(__MODULE__, args)
    defp cast(args), do: GenServer.cast(__MODULE__, args)

    # Callbacks
    def init(_) do
      {:ok, empty_state()}
    end

    def handle_call(:get, _from, state) do
      {:reply, state, state}
    end

    def handle_call(:reset, _from, _state) do
      {:reply, :ok, empty_state()}
    end

    def handle_call({:add_monitored_layer, name}, _from, state) do
      next_state =
        %{
          state
          | nb_monitored_layers: state.nb_monitored_layers + 1,
            monitored_layers: [name | state.monitored_layers]
        }
      {:reply, :ok, next_state}
    end

    def handle_call(:loop_started, _from, state) do
      {:reply, :ok, %{state | loop_started: true}}
    end

    def handle_cast({:put_lrs, lrs}, state) do
      {:noreply, %{state | lrs: lrs}}
    end

    def handle_cast({:append_activations_stats, layer_name, kernel}, state) do
      mean = %{
        type: :mean,
        value: Nx.to_number(Nx.mean(kernel)),
        layer: layer_name,
        iteration: state.iteration
      }

      std = %{
        type: :std,
        value: Nx.to_number(Nx.standard_deviation(kernel)),
        layer: layer_name,
        iteration: state.iteration
      }

      hist = %{
        type: :hist,
        value: NxExtra.histogram(Nx.abs(kernel), bins: 40, min: 0, max: 10),
        layer: layer_name,
        iteration: state.iteration
      }

      {next_layer, next_iteration} =
        if state.current_layer < state.nb_monitored_layers - 1 do
          {state.current_layer + 1, state.iteration}
        else
          {0, state.iteration + 1}
        end

      next_state = %{
        state
        | activations_stats: [mean, std, hist] ++ state.activations_stats,
          current_layer: next_layer,
          iteration: next_iteration
      }

      {:noreply, next_state}
    end
  end

  defmodule Plot do
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
        |> Enum.reverse() # layers in order

      hists = for l <- layers, do: hist(l)
      Kino.Layout.grid(hists, columns: 3)
    end

    def hist(layer, img_scale \\ 5) do
      hist_tensor =
        State.get(:activations_stats)
        |> Enum.filter(&(&1.type == :hist && &1.layer == layer))
        |> Enum.reverse() # iterations 0 is last so putting it back first
        |> Enum.map(&(&1.value))
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
        |> Nx.subtract(max) # max becomes 0 (darker)
        |> Nx.abs()
        |> Nx.reverse(axes: [0])
        |> Nx.multiply(scale)
        |> Nx.as_type(:u8)
        |> Nx.new_axis(-1) # adding channel dimension
        # upsampling
        |> Nx.new_axis(0)
        |> Axon.Layers.resize(size: {w, h}, channels: :last, method: :nearest)
        |> Nx.reshape({w, h, 1})
        |> Kino.Image.new()

      Kino.Layout.grid([Kino.Text.new(layer), img], columns: 1)
    end
  end
end
