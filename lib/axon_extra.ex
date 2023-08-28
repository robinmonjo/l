defmodule AxonExtra do
  def nodes_id_name_mapping(%Axon{} = model) do
    {mapping, _} =
      Axon.reduce_nodes(model, {%{}, %{}}, fn node, {mapping, counts} ->
        count = Map.get(counts, node.op, 0)

        {
          Map.put(mapping, node.id, "#{node.op}_#{count}"),
          Map.put(counts, node.op, count + 1)
        }
      end)

    mapping
  end
end
