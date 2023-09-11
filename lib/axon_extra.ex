defmodule AxonExtra do
  def nodes_id_name_mapping(%Axon{} = model) do
    {mapping, _} =
      Axon.reduce_nodes(model, {%{}, %{}}, fn node, {mapping, counts} ->
        count = Map.get(counts, node.op_name, 0)

        {
          Map.put(mapping, node.id, "#{node.op_name}_#{count}"),
          Map.put(counts, node.op_name, count + 1)
        }
      end)

    mapping
  end

  def relu(opts \\ []) do
    fn x, _ ->
      AxonExtra.Activations.relu(x, opts)
    end
  end
end
