defmodule L.State do
  alias L.State

  use GenServer

  defstruct [
    :loop_started,
    :activations_stats,
    :nb_monitored_layers,
    :monitored_layers,
    :current_layer,
    :iteration,
    :losses_for_lrs,
    :lrs
  ]

  def init() do
    case GenServer.start_link(__MODULE__, empty_state(), name: __MODULE__) do
      {:ok, _} ->
        :ok

      {:error, {:already_started, _}} ->
        # loop is already here, should we reset it ?
        case get() do
          %State{loop_started: true} ->
            # loop was started, the state in it is from previous training
            reset()
            :ok

          _ ->
            :ok
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
      losses_for_lrs: [],
      lrs: []
    }
  end

  # API
  def get(), do: call(:get)
  def get(key), do: Map.get(get(), key)

  def reset(), do: call(:reset)
  def add_monitored_layer(name), do: call({:add_monitored_layer, name})
  def loop_started(), do: call(:loop_started)
  def append_lr(lr), do: cast({:append_lr, lr})

  def put_losses_for_lrs(lrs), do: cast({:put_losses_for_lrs, lrs})

  def append_activations_stats(layer_name, kernel) do
    cast({:append_activations_stats, layer_name, kernel})
  end

  def stacked_histograms(layer), do: call({:stacked_histograms, layer})

  defp call(args), do: GenServer.call(__MODULE__, args)
  defp cast(args), do: GenServer.cast(__MODULE__, args)

  # Callbacks
  def init(_) do
    {:ok, empty_state()}
  end

  def handle_call(:get, _from, state) do
    {:reply, state, state}
  end

  def handle_call({:stacked_histograms, layer}, _from, state) do
    tensor =
      state.activations_stats
      |> Enum.filter(&(&1.type == :hist && &1.layer == layer))
      # iterations 0 is last so putting it back first
      |> Enum.reverse()
      |> Enum.map(& &1.value)
      |> Nx.stack()
      |> Nx.transpose()
      |> Nx.log1p()

    {:reply, tensor, state}
  end

  def handle_call(:reset, _from, _state) do
    {:reply, :ok, empty_state()}
  end

  def handle_call({:add_monitored_layer, name}, _from, state) do
    next_state = %{
      state
      | nb_monitored_layers: state.nb_monitored_layers + 1,
        monitored_layers: [name | state.monitored_layers]
    }

    {:reply, :ok, next_state}
  end

  def handle_call(:loop_started, _from, state) do
    {:reply, :ok, %{state | loop_started: true}}
  end

  def handle_cast({:put_losses_for_lrs, lrs}, state) do
    {:noreply, %{state | losses_for_lrs: lrs}}
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

  def handle_cast({:append_lr, lr}, %{lrs: lrs} = state) do
    point = %{
      iteration: state.iteration,
      lr: abs(Nx.to_number(lr)) # Axon optimizers return the scale so it might be negative
    }
    {:noreply, %{state | lrs: lrs ++ [point]}}
  end
end
