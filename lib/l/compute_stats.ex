defmodule L.ComputeStats do
  import Nx.Defn

  deftransformp axes(t) do
    [_ | axes] = Nx.axes(t)
    axes
  end

  def mean(t) do
    Nx.Defn.jit(&Nx.mean/2).(t, axes: axes(t))
  end

  def std(t) do
    Nx.Defn.jit(&Nx.standard_deviation/2).(t, axes: axes(t))
  end

  def hist(t) do
    EXLA.jit(&compute_hist/1).(t)
  end

  defnp compute_hist(tensors) do
    bins = 40
    size = elem(Nx.shape(tensors), 0)

    {_, result} =
      while {i = 0, acc = Nx.broadcast(1, {size, bins})}, t <- tensors do
        hist = NxExtra.histogram(Nx.abs(t), bins: bins, min: 0, max: 10)
        res = Nx.put_slice(acc, [i, 0], Nx.new_axis(hist, 0))
        {i + 1, res}
      end

    result
  end
end
