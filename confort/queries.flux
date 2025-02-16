import "strings"

from(bucket: "home_assistant_data")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "°C")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["domain"] == "sensor")
  |> filter(fn: (r) => 
      not strings.containsStr(v: r.entity_id, substr: "chip") and
      not strings.containsStr(v: r.entity_id, substr: "machine") and
      not strings.containsStr(v: r.entity_id, substr: "cpu")
  )
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")
  


import "strings"

from(bucket: "home_assistant_data")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "%")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["domain"] == "sensor")
  |> filter(fn: (r) => 
      strings.containsStr(v: r.entity_id, substr: "tuya_termometro_wifi_humidity") or 
      strings.containsStr(v: r.entity_id, substr: "shelly_blu_5480_humidity") or 
      strings.containsStr(v: r.entity_id, substr: "zigbee_sonoff_snzb02_01_humidity") or 
      strings.containsStr(v: r.entity_id, substr: "zigbee_heiman_hs3aq_humidity") 
  )
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")



import "strings"

from(bucket: "home_assistant_data")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "°C")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["domain"] == "sensor")
  |> filter(fn: (r) => 
        strings.containsStr(v: r.entity_id, substr: "tuya_termometro_wifi_temperature") or 
        strings.containsStr(v: r.entity_id, substr: "shelly_blu_5480_temperature") or 
        strings.containsStr(v: r.entity_id, substr: "zigbee_sonoff_snzb02_01_temperature") or 
        strings.containsStr(v: r.entity_id, substr: "zigbee_heiman_hs3aq_temperature") 
  )
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")


 