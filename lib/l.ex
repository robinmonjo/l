defmodule L do
  alias VegaLite, as: Vl
  alias L.Store
  alias Axon.Loop

  # Plot loss
  def plot_loss(%Loop{} = loop) do
    plot = render_loss_plot()

    loop
    |> Loop.kino_vega_lite_plot(plot, "loss")
  end

  defp render_loss_plot do
    plot = loss_plot()
    Kino.render(plot)
    plot
  end

  defp loss_plot do
    Vl.new(width: 600, height: 200)
    |> Vl.mark(:line)
    |> Vl.encode_field(:x, "step", type: :quantitative)
    |> Vl.encode_field(:y, "loss", type: :quantitative)
    |> Kino.VegaLite.new()
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
    Store.init()

    loop
    |> Loop.handle_event(:iteration_completed, fn state ->
      increase_learning_rate(state, lr_mult)
    end)
  end

  defp increase_learning_rate(%Loop.State{} = state, lr_mult) do
    loss = Nx.to_number(state.step_state.loss)
    prev_min_loss = Map.get(state.handler_metadata, :min_loss, 10_000)

    min_loss =
      if loss < prev_min_loss do
        loss
      else
        prev_min_loss
      end

    lrs = Map.get(state.handler_metadata, :lrs, [])

    if loss > 3 * prev_min_loss do
      Store.put_lrs(lrs)
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
        |> put_in([Access.key!(:step_state), Access.key!(:optimizer_state)], {%{scale: next_scale}})
        |> Map.put(:handler_metadata, next_metadata)

      {:continue, next_state}
    end
  end

  # plot activations stats
  def track_activations_stats(%Loop{} = loop) do
    Store.init()

    loop
    |> Loop.handle_event(:iteration_completed, &accumulate_activations_stats/1)
    |> Loop.handle_event(:epoch_completed, &store_activations_stats/1)
  end

  defp accumulate_activations_stats(%Loop.State{} = state) do
    stats = Map.get(state.handler_metadata, :activations_stats, [])
    iteration = Map.get(state.handler_metadata, :real_iteration, 0)

    IO.inspect(state)

    model_state = state.step_state.model_state
    next_stats = Enum.reduce(model_state, stats, fn({layer_name, value}, acc) ->
      kernel = value["kernel"]

      mean = %{
        type: :mean,
        value: Nx.to_number(Nx.mean(kernel)),
        layer: layer_name,
        iteration: iteration
      }

      std = %{
        type: :std,
        value: Nx.to_number(Nx.standard_deviation(kernel)),
        layer: layer_name,
        iteration: iteration
      }

      [mean | [std | acc]]
    end)

    next_metadata =
      state.handler_metadata
      |> Map.put(:activations_stats, next_stats)
      |> Map.put(:real_iteration, iteration + 1)

    {:continue, %{state | handler_metadata: next_metadata }}
  end

  defp store_activations_stats(%Loop.State{} = state) do
    stats = Map.get(state.handler_metadata, :activations_stats)
    Store.append_activations_stats(stats)
    next_metadata = Map.put(state.handler_metadata, :activations_stats, [])
    {:continue, %{state | handler_metadata: next_metadata }}
  end

  # Storage to access after a training
  defmodule Store do
    defstruct [
      :activations_stats,
      :lrs
    ]

    def init do
      if :ets.whereis(__MODULE__) != :undefined, do: :ets.delete(__MODULE__)
      :ets.new(__MODULE__, [:named_table, :public])

      store = %Store{
        activations_stats: []
      }
      save(store)
    end

    def get do
      [{:store, store}] = :ets.lookup(__MODULE__, :store)
      store
    end

    def put_lrs(lrs) do
      store = get()
      %{store | lrs: lrs}
      |> save()
    end

    def plot_lrs() do
      line_chart(get().lrs)
      |> Vl.encode_field(:x, "lr", type: :quantitative, scale: [type: :log])
      |> Vl.encode_field(:y, "loss", type: :quantitative)
      |> Kino.VegaLite.new()
    end

    def append_activations_stats(stats) do
      store = get()
      %{store | activations_stats: stats ++ store.activations_stats}
      |> save()
    end

    def plot_mean(layer \\ nil) do
      plot(:mean, layer)
    end

    def plot_std(layer \\ nil) do
      plot(:std, layer)
    end

    defp plot(type, layer) do
      %Store{activations_stats: stats} = get()

      stats = Enum.filter(stats, fn(stat) ->
        if layer do
          stat.type == type && stat.layer == layer
        else
          stat.type == type
        end
      end)

      line_chart(stats)
      |> Vl.encode_field(:x, "iteration", type: :quantitative)
      |> Vl.encode_field(:y, "value", type: :quantitative)
      |> Vl.encode_field(:color, "layer", type: :nominal)
      |> Kino.VegaLite.new()
    end

    defp line_chart(points) do
      Vl.new(width: 600, height: 200)
      |> Vl.mark(:line)
      |> Vl.data_from_values(points)
    end

    defp save(store) do
      true = :ets.insert(__MODULE__, {:store, store})
      store
    end
  end
end
