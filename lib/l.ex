defmodule L do
  alias L.{State, Plot}
  alias Axon.Loop

  def trainer(model, loss, optimizer, opts \\ []) do
    State.init()

    opts = Keyword.validate!(opts, monitored_layers: [])
    loop_trainer(model, loss, optimizer, opts)
  end

  defp loop_trainer(model, loss, optimizer, monitored_layers: []) do
    fun =
      if is_atom(loss) do
        &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
      else
        loss
      end

    model
    |> Loop.trainer(loss, optimizer, log: 1)
    |> Loop.handle_event(:started, fn state ->
      State.loop_started()
      {:continue, state}
    end)
    |> Loop.metric(fun, "iteration_loss", fn _, obs, _ -> obs end)
    |> kino_early_stop()
  end

  defp loop_trainer(model, loss, optimizer, monitored_layers: layers) do
    mapping = AxonExtra.nodes_id_name_mapping(model)

    Axon.map_nodes(model, fn %Axon.Node{id: id} = node ->
      name = mapping[id]

      if Enum.member?(layers, name) do
        fun = &State.append_layer_kernel(name, &1)
        Axon.attach_hook(node, fun, on: :forward)
      else
        node
      end
    end)
    |> loop_trainer(loss, optimizer, monitored_layers: [])
  end

  def fit(%Loop{} = loop, training_data, epochs, opts \\ []) do
    {time, res} =
      :timer.tc(fn ->
        options = [epochs: epochs, compiler: EXLA] ++ opts
        Loop.run(loop, training_data, %{}, options)
      end)

    IO.puts("\nTime: #{time / 1_000_000}s")
    res
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

  # LR finder
  def lr_finder_mode(%Loop{} = loop, lr_mult \\ 1.3) do
    loop
    |> Loop.handle_event(:iteration_completed, fn state ->
      increase_learning_rate(state, lr_mult)
    end)
  end

  defp increase_learning_rate(%Loop.State{} = state, lr_mult) do
    loss = Nx.to_number(state.metrics["iteration_loss"])
    prev_min_loss = get_metadata(state, :min_loss, 10_000)

    min_loss =
      if loss < prev_min_loss do
        loss
      else
        prev_min_loss
      end

    lrs = get_metadata(state, :lrs, [])

    if loss > 3 * prev_min_loss do
      State.put_losses_for_lrs(lrs)
      Kino.render(Plot.losses_for_lrs())
      {:halt_loop, state}
    else
      # sgd and adam optimizers lr are stored this way, might be different for other
      # optimizers
      opt_state = state.step_state.optimizer_state
      %{scale: scale} = elem(opt_state, 0)

      point = %{
        loss: loss,
        lr: abs(Nx.to_number(scale))
      }

      next_scale = Nx.multiply(scale, lr_mult)

      next_state =
        state
        |> put_metadata(:min_loss, min_loss)
        |> append_metadata(:lrs, point)
        |> put_in(
          [Access.key!(:step_state), Access.key!(:optimizer_state)],
          put_elem(opt_state, 0, %{scale: next_scale})
        )

      {:continue, next_state}
    end
  end

  def record_lr(%Loop{} = loop, opts \\ []) do
    loop
    |> Loop.handle_event(:iteration_completed, fn state ->
      lr =
        case elem(state.step_state.optimizer_state, 0) do
          %{scale: lr} ->
            lr

          %{count: step} ->
            scheduler = Keyword.fetch!(opts, :scheduler)
            scheduler.(step)
        end

      State.append_lr(lr)
      {:continue, state}
    end)
  end

  defp kino_early_stop(loop) do
    frame = Kino.Frame.new() |> Kino.render()
    stop_button = Kino.Control.button("stop")
    Kino.Frame.render(frame, stop_button)

    {:ok, button_agent} = Agent.start_link(fn -> nil end)

    stop_button
    |> Kino.Control.stream()
    |> Kino.listen(fn _event ->
      Agent.update(button_agent, fn _ -> :stop end)
    end)

    handler = fn state ->
      stop_state = Agent.get(button_agent, & &1)

      if stop_state == :stop do
        Agent.stop(button_agent)
        Kino.Frame.render(frame, "stopped")
        {:halt_loop, state}
      else
        {:continue, state}
      end
    end

    Loop.handle_event(loop, :iteration_completed, handler)
  end

  defp get_metadata(state, key, default \\ nil) do
    Map.get(state.handler_metadata, key, default)
  end

  defp put_metadata(state, key, value) do
    next_metadata =
      state.handler_metadata
      |> Map.put(key, value)

    %{state | handler_metadata: next_metadata}
  end

  defp append_metadata(state, key, value) do
    values = get_metadata(state, key, [])
    values = values ++ [value]
    put_metadata(state, key, values)
  end
end
