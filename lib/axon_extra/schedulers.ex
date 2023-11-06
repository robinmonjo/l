defmodule AxonExtra.Schedulers do
  import Nx.Defn

  def one_cycle_lr(opts \\ []) do
    &apply_one_cycle_lr(&1, opts)
  end

  defnp apply_one_cycle_lr(step, opts \\ []) do
    opts =
      keyword!(opts,
        initial_lr: 1.0e-4,
        max_lr: 0.6,
        steps: 1000,
        pct_start: 0.3,
        min_lr: 0.0
      )

    break = Nx.multiply(opts[:steps], opts[:pct_start])

    {cos, alpha} =
      if step < break do
        # "warmup" phase
        theta = Nx.Constants.pi() * step / break
        {1 - Nx.cos(theta), opts[:initial_lr]}
      else
        theta = Nx.Constants.pi() * (step - break) / (opts[:steps] - break)
        {1 + Nx.cos(theta), opts[:min_lr]}
      end

    cos
    |> Nx.divide(2)
    |> Nx.multiply(opts[:max_lr] - alpha)
    |> Nx.add(alpha)
  end
end
