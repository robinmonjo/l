defmodule L.HW do
  def inspect do
    EXLA.Client.get_supported_platforms()
    |> Map.keys()
    |> Enum.filter(&(&1 != :interpreter))
    |> Enum.each(fn name ->
      %EXLA.Client{device_count: count} = EXLA.Client.fetch!(name)
      IO.puts("#{name}, #{count}")
    end)
  end

  def set(client, device_id \\ 0)
  def set(:cpu, device_id), do: set(:host, device_id)
  def set(:gpu, device_id), do: set(:cuda, device_id)

  def set(client, device_id) do
    set_nx_backend(client: client, device_id: device_id)
    Nx.default_backend()
  end

  def compiler(client, device_id \\ 0)
  def compiler(:cpu, device_id \\ 0), do: compiler(:host, device_id)
  def compiler(:gpu, device_id \\ 0), do: compiler(:cuda, device_id)

  def compiler(client, device_id) do
    {EXLA.Backend, client: client, device_id: device_id}
  end

  defp set_nx_backend(opts) do
    Nx.default_backend({EXLA.Backend, opts})
  end
end
