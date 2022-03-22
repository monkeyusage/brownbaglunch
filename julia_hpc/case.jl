using DataFrames, CSV

df = CSV.read("couleur.tsv", DataFrame)

unique(df[!, "genre"])

replacements = Dict([item => counter for (counter, item) in enumerate(unique(df[!, "genre"]))]...)

df[!, "genre"] = map(x -> replacements[x], df[!, "genre"])

df[df[!, "genre"] .===2, :]

replace!(df[!, "genre"], missing => mean(filter(x -> x !== missing, df[!,"genre"])))
