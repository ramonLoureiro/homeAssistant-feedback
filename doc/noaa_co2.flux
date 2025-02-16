from(bucket: "ha_data")
  |> range(start: 1990-01-01T00:00:00Z)  // Establecer el inicio del rango a 1990
  |> filter(fn: (r) => r["_measurement"] == "co2_levels")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")

from(bucket: "ha_data")
  |> range(start: 1990-01-01T00:00:00Z)  // Ajusta la fecha de inicio según sea necesario
  |> filter(fn: (r) => r["_measurement"] == "co2_levels")
  |> filter(fn: (r) => r["_field"] == "average")  // Filtra solo el campo "average"
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")

from(bucket: "ha_data")
  |> range(start: 1990-01-01T00:00:00Z, stop: 2029-01-01T00:00:00Z)  // Rango hasta el futuro
  |> filter(fn: (r) => r["_measurement"] == "co2_levels")
  |> filter(fn: (r) => r["_field"] == "average")
  |> sort(columns: ["_time"], desc: false)
  |> yield(name: "average")

from(bucket: "NOAA_CO2")
  |> range(start: 1990-01-01T00:00:00Z, stop: 2035-01-01T00:00:00Z)  // Rango hasta el futuro
  |> filter(fn: (r) => r["_measurement"] == "co2_levels")
  |> filter(fn: (r) => r["_field"] == "average")
  |> sort(columns: ["_time"], desc: false)
  |> yield(name: "average")

  
from(bucket: "NOAA_CO2_LEVELS")
  |> range(start: 1990-01-01T00:00:00Z, stop: 2035-01-01T00:00:00Z)  // Rango hasta el futuro
  |> filter(fn: (r) => r["_measurement"] == "co2_levels")
  |> filter(fn: (r) => r["_field"] == "average")
  |> group()  
  |> count()
