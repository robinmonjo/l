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
      State.loop_started()
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
end
