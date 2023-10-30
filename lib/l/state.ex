defmodule L.State do
  alias L.{State, Layer}

  use GenServer

  defstruct [
    :loop_started,
    :layers,
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
      layers: %{},
      losses_for_lrs: [],
      lrs: []
    }
  end

  # API
  def get(), do: call(:get)
  def get(key), do: Map.get(get(), key)

  def reset(), do: call(:reset)
  def loop_started(), do: call(:loop_started)
  def append_lr(lr), do: cast({:append_lr, lr})
  def put_losses_for_lrs(lrs), do: cast({:put_losses_for_lrs, lrs})

  def append_layer_kernel(layer_name, kernel) do
    cast({:append_layer_kernel, layer_name, kernel})
  end

  def layer_stats(name, type), do: call({:layer_stats, name, type})

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

  def handle_call(:loop_started, _from, state) do
    {:reply, :ok, %{state | loop_started: true}}
  end

  def handle_cast({:put_losses_for_lrs, lrs}, state) do
    {:noreply, %{state | losses_for_lrs: lrs}}
  end

  def handle_cast({:append_layer_kernel, layer_name, kernel}, state) do
    layer = Map.get(state.layers, layer_name, %Layer{name: layer_name})
    layer = Layer.append(layer, kernel)
    next_layers = Map.put(state.layers, layer_name, layer)

    {:noreply, %{state | layers: next_layers}}
  end

  def handle_cast({:append_lr, lr}, %{lrs: lrs} = state) do
    lr_n = abs(Nx.to_number(lr)) # Axon optimizers return the scale so it might be negative
    {:noreply, %{state | lrs: [lr_n | lrs]}}
  end
end
