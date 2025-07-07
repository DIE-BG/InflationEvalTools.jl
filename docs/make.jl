using InflationEvalTools
using Documenter

# Add internal registry
import Pkg
Pkg.Registry.add(url = "https://github.com/DIE-BG/RegistryDIE.git")

DocMeta.setdocmeta!(InflationEvalTools, :DocTestSetup, :(using InflationEvalTools); recursive=true)

makedocs(;
    modules=[InflationEvalTools],
    authors="Rodrigo Chang <rrcp777@gmail.com> and contributors",
    sitename="InflationEvalTools.jl",
    format=Documenter.HTML(;
        canonical="https://DIE-BG.github.io/InflationEvalTools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DIE-BG/InflationEvalTools.jl",
    devbranch="main",
)
