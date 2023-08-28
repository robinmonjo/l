defmodule L.MixProject do
  use Mix.Project

  def project do
    [
      app: :l,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.6.0"},
      {:kino_vega_lite, "~> 0.1.7"},
      {:axon, "~> 0.6.0"}
    ]
  end
end
