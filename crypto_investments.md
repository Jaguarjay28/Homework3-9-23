# Module 10 Application

## Challenge: Crypto Clustering

In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.

The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.

The steps for this challenge are broken out into the following sections:

* Import the Data (provided in the starter code)
* Prepare the Data (provided in the starter code)
* Find the Best Value for `k` Using the Original Data
* Cluster Cryptocurrencies with K-means Using the Original Data
* Optimize Clusters with Principal Component Analysis
* Find the Best Value for `k` Using the PCA Data
* Cluster the Cryptocurrencies with K-means Using the PCA Data
* Visualize and Compare the Results

### Import the Data

This section imports the data into a new DataFrame. It follows these steps:

1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.

2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.


> **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 


```python
# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```


```python
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>1.08388</td>
      <td>7.60278</td>
      <td>6.57509</td>
      <td>7.67258</td>
      <td>-3.25185</td>
      <td>83.51840</td>
      <td>37.51761</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.22392</td>
      <td>10.38134</td>
      <td>4.80849</td>
      <td>0.13169</td>
      <td>-12.88890</td>
      <td>186.77418</td>
      <td>101.96023</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.21173</td>
      <td>0.04935</td>
      <td>0.00640</td>
      <td>-0.04237</td>
      <td>0.28037</td>
      <td>-0.00542</td>
      <td>0.01954</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.37819</td>
      <td>-0.60926</td>
      <td>2.24984</td>
      <td>0.23455</td>
      <td>-17.55245</td>
      <td>39.53888</td>
      <td>-16.60193</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>2.90585</td>
      <td>17.09717</td>
      <td>14.75334</td>
      <td>15.74903</td>
      <td>-13.71793</td>
      <td>21.66042</td>
      <td>14.49384</td>
    </tr>
    <tr>
      <th>binancecoin</th>
      <td>2.10423</td>
      <td>12.85511</td>
      <td>6.80688</td>
      <td>0.05865</td>
      <td>36.33486</td>
      <td>155.61937</td>
      <td>69.69195</td>
    </tr>
    <tr>
      <th>chainlink</th>
      <td>-0.23935</td>
      <td>20.69459</td>
      <td>9.30098</td>
      <td>-11.21747</td>
      <td>-43.69522</td>
      <td>403.22917</td>
      <td>325.13186</td>
    </tr>
    <tr>
      <th>cardano</th>
      <td>0.00322</td>
      <td>13.99302</td>
      <td>5.55476</td>
      <td>10.10553</td>
      <td>-22.84776</td>
      <td>264.51418</td>
      <td>156.09756</td>
    </tr>
    <tr>
      <th>litecoin</th>
      <td>-0.06341</td>
      <td>6.60221</td>
      <td>7.28931</td>
      <td>1.21662</td>
      <td>-17.23960</td>
      <td>27.49919</td>
      <td>-12.66408</td>
    </tr>
    <tr>
      <th>bitcoin-cash-sv</th>
      <td>0.92530</td>
      <td>3.29641</td>
      <td>-1.86656</td>
      <td>2.88926</td>
      <td>-24.87434</td>
      <td>7.42562</td>
      <td>93.73082</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate summary statistics
df_market_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.269686</td>
      <td>4.497147</td>
      <td>0.185787</td>
      <td>1.545693</td>
      <td>-0.094119</td>
      <td>236.537432</td>
      <td>347.667956</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.694793</td>
      <td>6.375218</td>
      <td>8.376939</td>
      <td>26.344218</td>
      <td>47.365803</td>
      <td>435.225304</td>
      <td>1247.842884</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-13.527860</td>
      <td>-6.094560</td>
      <td>-18.158900</td>
      <td>-34.705480</td>
      <td>-44.822480</td>
      <td>-0.392100</td>
      <td>-17.567530</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.608970</td>
      <td>0.047260</td>
      <td>-5.026620</td>
      <td>-10.438470</td>
      <td>-25.907990</td>
      <td>21.660420</td>
      <td>0.406170</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.063410</td>
      <td>3.296410</td>
      <td>0.109740</td>
      <td>-0.042370</td>
      <td>-7.544550</td>
      <td>83.905200</td>
      <td>69.691950</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.612090</td>
      <td>7.602780</td>
      <td>5.510740</td>
      <td>4.578130</td>
      <td>0.657260</td>
      <td>216.177610</td>
      <td>168.372510</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.840330</td>
      <td>20.694590</td>
      <td>24.239190</td>
      <td>140.795700</td>
      <td>223.064370</td>
      <td>2227.927820</td>
      <td>7852.089700</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)
```






<div id='1002'>
  <div class="bk-root" id="8fcc90b1-1f7c-49ad-bfb5-e7601e35b0bc" data-root-id="1002"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"e27fbd9b-4f85-4ca7-a50f-6c1528051471":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{},"id":"1110","type":"UnionRenderers"},{"attributes":{},"id":"1164","type":"UnionRenderers"},{"attributes":{},"id":"1023","type":"CategoricalTicker"},{"attributes":{"data":{"Variable":["price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"xY8xdy3hVED0piIVxlhnQBe86CtIM3a/9zsUBfrEQ0DTn/1IEak1QPTDCOHRc2NAl3MprqozeUAexM4UOohwQIB9dOrKfztAFhiyutWzHUCN7iB2plZsQIQqNXugFcS/ZHWr56QPMkALe9rhrzBlQH+8V61MnGBA5nlwd9a2RUARHm0csfpDQC+Lic3HJ1VAkj8YeO4/Z0AOvjCZKhjZv2vUQzS61GNAtRX7y+75VEC5GW7A52cwQAPso1NXHkVAKa4q+64Fa0A7NgLxun65P+TaUDHOwVdAyM1wAz7bcEAFwHgGDaNeQH3Qs1k1lYtAOUVHcvnvGEANGvonuMgkQCYZOQvbZ6FAGD4ipkSiVED5MeauZXeDQHi0ccQaSoJAC170FcTYmEB1PGagMmJLQC2yne+nvkxAndfYJaq3tr9R9wFIbSl+QA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1170"},"selection_policy":{"id":"1194"}},"id":"1169","type":"ColumnDataSource"},{"attributes":{"coordinates":null,"data_source":{"id":"1141"},"glyph":{"id":"1144"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1146"},"nonselection_glyph":{"id":"1145"},"selection_glyph":{"id":"1168"},"view":{"id":"1148"}},"id":"1147","type":"GlyphRenderer"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1068","type":"Line"},{"attributes":{"axis_label":"coin_id","coordinates":null,"formatter":{"id":"1043"},"group":null,"major_label_orientation":1.5707963267948966,"major_label_policy":{"id":"1044"},"ticker":{"id":"1023"}},"id":"1022","type":"CategoricalAxis"},{"attributes":{},"id":"1047","type":"AllLabels"},{"attributes":{"label":{"value":"price_change_percentage_60d"},"renderers":[{"id":"1147"}]},"id":"1167","type":"LegendItem"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"1067"},{"id":"1089"},{"id":"1113"},{"id":"1139"},{"id":"1167"},{"id":"1197"},{"id":"1229"}],"location":[0,0],"title":"Variable"},"id":"1066","type":"Legend"},{"attributes":{"label":{"value":"price_change_percentage_30d"},"renderers":[{"id":"1121"}]},"id":"1139","type":"LegendItem"},{"attributes":{"line_alpha":0.1,"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1073","type":"Line"},{"attributes":{"source":{"id":"1141"}},"id":"1148","type":"CDSView"},{"attributes":{},"id":"1116","type":"Selection"},{"attributes":{"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1198","type":"Line"},{"attributes":{"callback":null,"renderers":[{"id":"1054"},{"id":"1075"},{"id":"1097"},{"id":"1121"},{"id":"1147"},{"id":"1175"},{"id":"1205"}],"tags":["hv_created"],"tooltips":[["Variable","@{Variable}"],["coin_id","@{coin_id}"],["value","@{value}"]]},"id":"1006","type":"HoverTool"},{"attributes":{},"id":"1046","type":"BasicTickFormatter"},{"attributes":{"label":{"value":"price_change_percentage_1y"},"renderers":[{"id":"1205"}]},"id":"1229","type":"LegendItem"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01002","sizing_mode":"stretch_width"},"id":"1003","type":"Spacer"},{"attributes":{"line_alpha":0.1,"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1203","type":"Line"},{"attributes":{"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1144","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1145","type":"Line"},{"attributes":{},"id":"1043","type":"CategoricalTickFormatter"},{"attributes":{"coordinates":null,"data_source":{"id":"1115"},"glyph":{"id":"1118"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1120"},"nonselection_glyph":{"id":"1119"},"selection_glyph":{"id":"1140"},"view":{"id":"1122"}},"id":"1121","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.2,"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1146","type":"Line"},{"attributes":{"below":[{"id":"1022"}],"center":[{"id":"1024"},{"id":"1028"}],"height":400,"left":[{"id":"1025"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1054"},{"id":"1075"},{"id":"1097"},{"id":"1121"},{"id":"1147"},{"id":"1175"},{"id":"1205"}],"right":[{"id":"1066"}],"sizing_mode":"fixed","title":{"id":"1014"},"toolbar":{"id":"1035"},"width":800,"x_range":{"id":"1004"},"x_scale":{"id":"1018"},"y_range":{"id":"1005"},"y_scale":{"id":"1020"}},"id":"1013","subtype":"Figure","type":"Plot"},{"attributes":{"children":[{"id":"1003"},{"id":"1013"},{"id":"1253"}],"margin":[0,0,0,0],"name":"Row00998","tags":["embedded"]},"id":"1002","type":"Row"},{"attributes":{"axis":{"id":"1025"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1028","type":"Grid"},{"attributes":{"coordinates":null,"data_source":{"id":"1069"},"glyph":{"id":"1072"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1074"},"nonselection_glyph":{"id":"1073"},"selection_glyph":{"id":"1090"},"view":{"id":"1076"}},"id":"1075","type":"GlyphRenderer"},{"attributes":{"source":{"id":"1115"}},"id":"1122","type":"CDSView"},{"attributes":{},"id":"1226","type":"UnionRenderers"},{"attributes":{"label":{"value":"price_change_percentage_7d"},"renderers":[{"id":"1075"}]},"id":"1089","type":"LegendItem"},{"attributes":{"source":{"id":"1069"}},"id":"1076","type":"CDSView"},{"attributes":{"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1230","type":"Line"},{"attributes":{"coordinates":null,"data_source":{"id":"1199"},"glyph":{"id":"1202"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1204"},"nonselection_glyph":{"id":"1203"},"selection_glyph":{"id":"1230"},"view":{"id":"1206"}},"id":"1205","type":"GlyphRenderer"},{"attributes":{"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1114","type":"Line"},{"attributes":{},"id":"1030","type":"PanTool"},{"attributes":{"axis":{"id":"1022"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1024","type":"Grid"},{"attributes":{"data":{"Variable":["price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"VMa/z7iwHkAl6ZrJN9vAP/28qUiFsaW/L26jAbwFzj8dcjPcgH8vQPMf0m9fB64/5bhTOlhvJsBnfjUHCDYkQGWNeohGd/M/sCDNWDQdB0Cmft5UpHIxwJgvL8A+OsW/t0WZDTKJEsDeVKTC2HpDQEzD8BExNSnAjWK5pdVALMDjjcwjf+AkwIrIsIo3QiDAjliLTwFQEkAWpBmLpnMBwE3WqIdo9DXAVG8NbJWAHUAnMQisHJoIQGA8g4b+CQrACcTr+gU7DcCPpQ9dUN+SP80Bgjl6nBDA9S1zuixWLMBC7Eyh83odQEku/yH9bj9AtTf4wmQqH8Am/FI/byoHQNjYJaq3/j3A9GxWfa62MECdRloqb9c0wF4R/G8lOyLArK3YX3aZYUC1/SsrTVpBwP/PYb68ICXAFR3J5T+knz+ndLD+z4EqQA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1116"},"selection_policy":{"id":"1136"}},"id":"1115","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.2,"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1074","type":"Line"},{"attributes":{"source":{"id":"1199"}},"id":"1206","type":"CDSView"},{"attributes":{},"id":"1070","type":"Selection"},{"attributes":{"line_alpha":0.2,"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1120","type":"Line"},{"attributes":{},"id":"1086","type":"UnionRenderers"},{"attributes":{"line_alpha":0.2,"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1204","type":"Line"},{"attributes":{},"id":"1200","type":"Selection"},{"attributes":{},"id":"1044","type":"AllLabels"},{"attributes":{},"id":"1142","type":"Selection"},{"attributes":{"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1072","type":"Line"},{"attributes":{},"id":"1063","type":"UnionRenderers"},{"attributes":{"tools":[{"id":"1006"},{"id":"1029"},{"id":"1030"},{"id":"1031"},{"id":"1032"},{"id":"1033"}]},"id":"1035","type":"Toolbar"},{"attributes":{"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1202","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1052","type":"Line"},{"attributes":{"data":{"Variable":["price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"kQpjC0HCQkDWqIdodH1ZQN/42jNLApQ/NXugFRiaMMDN6bKY2PwsQOcdp+hIbFFAaTo7GRxSdEC6LCY2H4NjQPnaM0sCVCnA0NA/wcVuV0CQvd798RBzQL4wmSoYlci/t39lpUmRMcDovMYuUaFhQEPKT6p9nk1A6PaSxmiAYUBdUN8yp75hQMzuycNCnStAw7ZFmQ2cVEAHsTOFzmvSvx/0bFZ9aWBAhhvw+WHEQkAN/RNcrAA1QN8Vwf9WkjhAS7A4nPkWaUAVUn5S7dPBPzUk7rH04Q/AIsMq3shbaUBf0hito7hUQPyMCwcC64VAzsKedvirA8Dwoq8gzTgmQHRGlPYWrL5Ayk+qfTpOJUDfiVkvRjaFQEfJq3MMC3RAnFCIgONmn0CsVib8Uk8pwNttF5rrC2VAyXGndLD+2T/ZfFwbKgxnQA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1200"},"selection_policy":{"id":"1226"}},"id":"1199","type":"ColumnDataSource"},{"attributes":{"coordinates":null,"group":null,"text_color":"black","text_font_size":"12pt"},"id":"1014","type":"Title"},{"attributes":{"axis_label":"","coordinates":null,"formatter":{"id":"1046"},"group":null,"major_label_policy":{"id":"1047"},"ticker":{"id":"1026"}},"id":"1025","type":"LinearAxis"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1053","type":"Line"},{"attributes":{"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1168","type":"Line"},{"attributes":{"coordinates":null,"data_source":{"id":"1048"},"glyph":{"id":"1051"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1053"},"nonselection_glyph":{"id":"1052"},"selection_glyph":{"id":"1068"},"view":{"id":"1055"}},"id":"1054","type":"GlyphRenderer"},{"attributes":{},"id":"1194","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1119","type":"Line"},{"attributes":{},"id":"1033","type":"ResetTool"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1051","type":"Line"},{"attributes":{},"id":"1020","type":"LinearScale"},{"attributes":{"label":{"value":"price_change_percentage_200d"},"renderers":[{"id":"1175"}]},"id":"1197","type":"LegendItem"},{"attributes":{},"id":"1136","type":"UnionRenderers"},{"attributes":{"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1118","type":"Line"},{"attributes":{"coordinates":null,"data_source":{"id":"1169"},"glyph":{"id":"1172"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1174"},"nonselection_glyph":{"id":"1173"},"selection_glyph":{"id":"1198"},"view":{"id":"1176"}},"id":"1175","type":"GlyphRenderer"},{"attributes":{"factors":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"tags":[[["coin_id","coin_id",null]]]},"id":"1004","type":"FactorRange"},{"attributes":{"line_alpha":0.1,"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1095","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1173","type":"Line"},{"attributes":{"source":{"id":"1169"}},"id":"1176","type":"CDSView"},{"attributes":{"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1090","type":"Line"},{"attributes":{},"id":"1018","type":"CategoricalScale"},{"attributes":{"line_alpha":0.2,"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1174","type":"Line"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01003","sizing_mode":"stretch_width"},"id":"1253","type":"Spacer"},{"attributes":{},"id":"1170","type":"Selection"},{"attributes":{},"id":"1031","type":"WheelZoomTool"},{"attributes":{"data":{"Variable":["price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"zQaZZORMGkB5knTN5DsTQC1DHOviNno/ctwpHaz/AUCLic3HtYEtQJgvL8A+OhtACoDxDBqaIkAOhGQBEzgWQICfceFAKB1AIVnABG7d/b8xsfm4NhQhwGtI3GPpQ3e/pfeNrz0z8j8ao3VUNWkyQBUA4xk0dPY/VIzzN6GQAkCXrfVFQhsUwKpla32RUPG/Qgkzbf8KFkDPg7uzdtvFP8MN+PwwsiPAYFlpUgo6GkB3+GuyRh0cwM2v5gDBHN0/jliLTwEwBUDWrZ6T3jeuP7pOIy2VNw7Am1Wfq60oMsAJM23/yioXQJfK2xFOSxrAQE0tW+uL4D9OucK7XET0v0SLbOf7mSPA/pqsUQ/R978vaYzWUZUqwDtT6LzGbirAvodLjjs9OEByUMJM2x8lwJ30vvG1Z8q/ZCMQr+sXvD+7D0BqE0cdwA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1092"},"selection_policy":{"id":"1110"}},"id":"1091","type":"ColumnDataSource"},{"attributes":{},"id":"1026","type":"BasicTicker"},{"attributes":{"coordinates":null,"data_source":{"id":"1091"},"glyph":{"id":"1094"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1096"},"nonselection_glyph":{"id":"1095"},"selection_glyph":{"id":"1114"},"view":{"id":"1098"}},"id":"1097","type":"GlyphRenderer"},{"attributes":{"source":{"id":"1048"}},"id":"1055","type":"CDSView"},{"attributes":{"end":8641.780918,"reset_end":8641.780918,"reset_start":-834.5136980000001,"start":-834.5136980000001,"tags":[[["value","value",null]]]},"id":"1005","type":"Range1d"},{"attributes":{},"id":"1029","type":"SaveTool"},{"attributes":{"label":{"value":"price_change_percentage_24h"},"renderers":[{"id":"1054"}]},"id":"1067","type":"LegendItem"},{"attributes":{"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1172","type":"Line"},{"attributes":{"data":{"Variable":["price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"nZ0MjpJX8T8MzXUaaanMP5mByvj3Gcu/wCZr1EM02L8ep+hILj8HQL0Yyol21QBAwhcmUwWjzr//CS5W1GBqP7JGPUSjO7C/QKTfvg6c7T8urYbEPZbjPzeJQWDl0Ma/1pC4x9KHwj87NgLxun7bP90HILWJk7M/JjYf14aK5b83iUFg5dAFwG3i5H6HIvC/5nlwd9Zu7r9i83FtqBjLP+iHEcKjjd8/M9yAzw+j8T/fGtgqweLAv+RmuAGfH9q/UdobfGEy678D7KNTVz67v8YzaOif4No/teBFX0Ga9D8G2Eenrnzjvx+duvJZPhLAoMN8eQH28L+cxCCwcmjdv667eapDDivAX5hMFYxK479EUaBP5EkQwAZkr3d/XBNA0JuKVBgbBEDqBDQRNrz1vxo09E9wseo/tI6qJoi6r79qMA3DR8QHQA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1049"},"selection_policy":{"id":"1063"}},"id":"1048","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"1034"}},"id":"1032","type":"BoxZoomTool"},{"attributes":{"data":{"Variable":["price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"FvvL7skDCsAs1JrmHccpwKjjMQOV8dE/2qz6XG2NMcBzol2FlG8rwIleRrHcKkJAN8MN+PzYRcBi26LMBtk2wCEf9GxWPTHAq5UJv9TfOMBg5dAi23kwwDygbMoV3qU/l631RUIrPsDNzMzMzNxEQJm7lpAPGhZAms5OBkdpRsBKe4MvTKYGwEsfuqC+zT7APL1SliHOHMC4AZ8fRgjlP9L7xteeCStAFqQZi6azDMCvJeSDni0ewDeOWItPQQvAXvQVpBm3VEDaOGItPgW4PwpLPKBsQkHAaw4QzNHPRcAJM23/yoo0wBmQvd79AVRAfa62Yn85OsBVGFsIcug5wLCsNCkFHQFA529CIQKeMsA0uoPYmbZDwPKwUGua0VNA/pqsUQ/ia0Ao8iTpmllAwFuxv+yenBdAmrFoOjsZ0D8s1JrmHRc/wA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1142"},"selection_policy":{"id":"1164"}},"id":"1141","type":"ColumnDataSource"},{"attributes":{},"id":"1049","type":"Selection"},{"attributes":{"source":{"id":"1091"}},"id":"1098","type":"CDSView"},{"attributes":{},"id":"1092","type":"Selection"},{"attributes":{"label":{"value":"price_change_percentage_14d"},"renderers":[{"id":"1097"}]},"id":"1113","type":"LegendItem"},{"attributes":{"line_alpha":0.2,"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1096","type":"Line"},{"attributes":{"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1140","type":"Line"},{"attributes":{"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1094","type":"Line"},{"attributes":{"data":{"Variable":["price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"RbsKKT9pHkAzUBn/PsMkQL99HThnRKk/yVnY0w5/47/+JhQi4BgxQBL3WPrQtSlA7yB2ptCxNEA/jBAebfwrQPFL/bypaBpArKjBNAxfCkByv0NRoK8WwPlJtU/HY76/lPsdigL99L+n6Egu/5EvQE1KQbeX9BBA0m9fB87pIUBlU67wLjcSwKbtX1lpkgBAZwqd19gVMEBcIEHxY8ytP3wnZr0YigNAOh4zUBmfHUD3Hi457pT1vzNQGf8+4/k/kQ96Nqs+6781Y9F0djKoP2dEaW/wRRhAgXhdv2A3/L/8GHPXErIkQP8JLlbUYBjAVU0QdR9gFEDRlnMprmoIQGCrBIvD2RBAAiuHFtlOIECbG9MTllgRQKhXyjLEURtAqn06HjNQ4z+UvDrHgGzzvwltOZfiahxAsD2zJEBNxT8yj/zBwHPlPw==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1070"},"selection_policy":{"id":"1086"}},"id":"1069","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1034","type":"BoxAnnotation"}],"root_ids":["1002"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"e27fbd9b-4f85-4ca7-a50f-6c1528051471","root_ids":["1002"],"roots":{"1002":"8fcc90b1-1f7c-49ad-bfb5-e7601e35b0bc"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Prepare the Data

This section prepares the data before running the K-Means algorithm. It follows these steps:

1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.

2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.



```python
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)
```


```python
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
    </tr>
  </tbody>
</table>
</div>



---

### Find the Best Value for k Using the Original Data

In this section, you will use the elbow method to find the best value for `k`.

1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following question: What is the best value for `k`?


```python
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
```


```python
# Create an empy list to store the inertia values
inertia = []
```


```python
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=1, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)
```


```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)
df_elbow
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>inertia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>287.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>287.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot = df_elbow.hvplot.line(x="k" , y="inertia", title="Elbow Curve", xticks=k)
elbow_plot
```






<div id='1365'>
  <div class="bk-root" id="0c0fb95c-9b0e-4d0a-a65c-1982a273a46c" data-root-id="1365"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"c71b6097-6ffd-4d50-8b19-85ced4256000":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"axis":{"id":"1383"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1386","type":"Grid"},{"attributes":{},"id":"1389","type":"WheelZoomTool"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"1416"},"group":null,"major_label_policy":{"id":"1417"},"ticker":{"id":"1384"}},"id":"1383","type":"LinearAxis"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"1411"},"group":null,"major_label_policy":{"id":"1412"},"ticker":{"id":"1409"}},"id":"1379","type":"LinearAxis"},{"attributes":{"tools":[{"id":"1369"},{"id":"1387"},{"id":"1388"},{"id":"1389"},{"id":"1390"},{"id":"1391"}]},"id":"1393","type":"Toolbar"},{"attributes":{"below":[{"id":"1379"}],"center":[{"id":"1382"},{"id":"1386"}],"height":300,"left":[{"id":"1383"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1406"}],"sizing_mode":"fixed","title":{"id":"1371"},"toolbar":{"id":"1393"},"width":700,"x_range":{"id":"1367"},"x_scale":{"id":"1375"},"y_range":{"id":"1368"},"y_scale":{"id":"1377"}},"id":"1370","subtype":"Figure","type":"Plot"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1403","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1404","type":"Line"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1392","type":"BoxAnnotation"},{"attributes":{},"id":"1416","type":"BasicTickFormatter"},{"attributes":{},"id":"1401","type":"Selection"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1405","type":"Line"},{"attributes":{},"id":"1388","type":"PanTool"},{"attributes":{"axis":{"id":"1379"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1382","type":"Grid"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01208","sizing_mode":"stretch_width"},"id":"1366","type":"Spacer"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1408","type":"Line"},{"attributes":{},"id":"1411","type":"BasicTickFormatter"},{"attributes":{},"id":"1387","type":"SaveTool"},{"attributes":{"overlay":{"id":"1392"}},"id":"1390","type":"BoxZoomTool"},{"attributes":{},"id":"1375","type":"LinearScale"},{"attributes":{"source":{"id":"1400"}},"id":"1407","type":"CDSView"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"1409","type":"FixedTicker"},{"attributes":{"children":[{"id":"1366"},{"id":"1370"},{"id":"1427"}],"margin":[0,0,0,0],"name":"Row01204","tags":["embedded"]},"id":"1365","type":"Row"},{"attributes":{"end":288.2,"reset_end":288.2,"reset_start":285.8,"start":285.8,"tags":[[["inertia","inertia",null]]]},"id":"1368","type":"Range1d"},{"attributes":{"coordinates":null,"data_source":{"id":"1400"},"glyph":{"id":"1403"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1405"},"nonselection_glyph":{"id":"1404"},"selection_glyph":{"id":"1408"},"view":{"id":"1407"}},"id":"1406","type":"GlyphRenderer"},{"attributes":{},"id":"1384","type":"BasicTicker"},{"attributes":{},"id":"1412","type":"AllLabels"},{"attributes":{},"id":"1391","type":"ResetTool"},{"attributes":{},"id":"1424","type":"UnionRenderers"},{"attributes":{},"id":"1377","type":"LinearScale"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"1371","type":"Title"},{"attributes":{},"id":"1417","type":"AllLabels"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"1367","type":"Range1d"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01209","sizing_mode":"stretch_width"},"id":"1427","type":"Spacer"},{"attributes":{"data":{"inertia":{"__ndarray__":"AAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"1401"},"selection_policy":{"id":"1424"}},"id":"1400","type":"ColumnDataSource"},{"attributes":{"callback":null,"renderers":[{"id":"1406"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"1369","type":"HoverTool"}],"root_ids":["1365"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"c71b6097-6ffd-4d50-8b19-85ced4256000","root_ids":["1365"],"roots":{"1365":"0c0fb95c-9b0e-4d0a-a65c-1982a273a46c"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following question: What is the best value for k?
**Question:** What is the best value for `k`?

**Answer:** 4

---

### Cluster Cryptocurrencies with K-means Using the Original Data

In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the original data.

3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.

4. Create a copy of the original data and add a new column with the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.


```python
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)
```


```python
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)
```




    KMeans(n_clusters=1, random_state=0)




```python
# Predict the clusters to group the cryptocurrencies using the scaled data
crypto_cluster = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
print(crypto_cluster)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0]



```python
# Create a copy of the DataFrame
df_market_data_scaled_predictions = df_market_data_scaled.copy()
```


```python
# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_predictions["crypto_cluster"] = crypto_cluster

# Display sample data
df_market_data_scaled_predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
      <th>crypto_cluster</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
      <td>0</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
clusters_plot = df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="crypto_cluster",
    hover_cols=["coin_id"],
    marker=["hex", "square", "cross", "inverted_triangle"],
    title="Cryptocurrencies Clusters",  
)    
clusters_plot    
```






<div id='2096'>
  <div class="bk-root" id="00217166-eb1f-4013-a427-86f3961e5566" data-root-id="2096"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"7efd9d04-1772-4ed2-8989-aa065866e2bd":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"2141","type":"Scatter"},{"attributes":{"end":2.2155632386560065,"reset_end":2.2155632386560065,"reset_start":-5.276792781891412,"start":-5.276792781891412,"tags":[[["price_change_percentage_24h","price_change_percentage_24h",null]]]},"id":"2098","type":"Range1d"},{"attributes":{"data":{"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"crypto_cluster":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"price_change_percentage_24h":{"__ndarray__":"DdlqYN9F4D86azDosLzHP5giRdTpS5Y/5iURtxHfpL8wCyMArRbzP/cOjHA0iuw/1QxzqW9Xhz8zBL2hZD+6P0nbN/ba1rM/WpkZGJ+73D/+slcesDPVP/IxOp6elqE/H2STd0ruwz9Epj8/c9DQPxk0XoR2pcA/jEB40RVnw790fnl54Irtv9XPDahDw9G/qq8st/Fh0L/5U5ZBISbHP6Slu6TDVtI/BVlpN5h+4D+njODODoCqPwXDDo/fo6q/woaukuXmy79od0yHwmevP8HhEPEMldA/ypmU5gu54j+sKNPN2lDAv1YH9WGFy/m/N7JuVBEB07/EU4GAhEGyv91nhz6W7BPA/ACFOIMFwL/DnH9IcNv2v5azwSWNt/4/VLqXBn668D/IKcQzxy3av5WrmDyeito/Bf22eE/6sz9Y362Ir3rzPw==","dtype":"float64","order":"little","shape":[41]},"price_change_percentage_7d":{"__ndarray__":"U1k8q3mQ3z9yoPpI+ebtP22UiL5Pmua/Ubo8ah/z6b8wAdnT9gEAQIJpDjGZPPU/BTbpSPiTBEAmUWkixiD4P/OMI2gdZdU/DZTtp1doyL936vtUZtb5v0QZLH/ydOe/it8GjguF7b8rMp3jf678PwRAcFxBAKW/ZWZRGoqp5j8Udj8VbP/2vySelkVDp9i/XK6ZksNx/T/tJm9WzI7mvy674lUf4tS/Q3zKstaO3T84jfoETLXtv6NqpXs9Q92/wrTyyiEu678mI2rMB53mv1Euo6Hk788/3UZrgr7Q778kzwwdV7vtP/9TV66U6fq/ZWX+mWVCuD8GJ0pHuV/Nv3SFotqbIae/srW7TleV4j/JyBNmVh+av008Y/5ntdc/Nv03JFjJ47+Htdvf1gXtv2Avihmzf9o/e31xtwIC5r/STlu6Y3Ljvw==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"2138"},"selection_policy":{"id":"2152"}},"id":"2137","type":"ColumnDataSource"},{"attributes":{},"id":"2138","type":"Selection"},{"attributes":{"coordinates":null,"group":null,"text":"Cryptocurrencies Clusters","text_color":"black","text_font_size":"12pt"},"id":"2102","type":"Title"},{"attributes":{"coordinates":null,"data_source":{"id":"2137"},"glyph":{"id":"2140"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2142"},"nonselection_glyph":{"id":"2141"},"selection_glyph":{"id":"2157"},"view":{"id":"2144"}},"id":"2143","type":"GlyphRenderer"},{"attributes":{},"id":"2136","type":"AllLabels"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02150","sizing_mode":"stretch_width"},"id":"2168","type":"Spacer"},{"attributes":{},"id":"2152","type":"UnionRenderers"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"2156"}],"location":[0,0],"title":"crypto_cluster"},"id":"2155","type":"Legend"},{"attributes":{"source":{"id":"2137"}},"id":"2144","type":"CDSView"},{"attributes":{"callback":null,"renderers":[{"id":"2143"}],"tags":["hv_created"],"tooltips":[["crypto_cluster","@{crypto_cluster}"],["price_change_percentage_24h","@{price_change_percentage_24h}"],["price_change_percentage_7d","@{price_change_percentage_7d}"],["coin_id","@{coin_id}"]]},"id":"2100","type":"HoverTool"},{"attributes":{},"id":"2106","type":"LinearScale"},{"attributes":{},"id":"2135","type":"BasicTickFormatter"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"2143"}]},"id":"2156","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"2142","type":"Scatter"},{"attributes":{},"id":"2119","type":"PanTool"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"2157","type":"Scatter"},{"attributes":{"tools":[{"id":"2100"},{"id":"2118"},{"id":"2119"},{"id":"2120"},{"id":"2121"},{"id":"2122"}]},"id":"2124","type":"Toolbar"},{"attributes":{"below":[{"id":"2110"}],"center":[{"id":"2113"},{"id":"2117"}],"height":300,"left":[{"id":"2114"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2143"}],"right":[{"id":"2155"}],"sizing_mode":"fixed","title":{"id":"2102"},"toolbar":{"id":"2124"},"width":700,"x_range":{"id":"2098"},"x_scale":{"id":"2106"},"y_range":{"id":"2099"},"y_scale":{"id":"2108"}},"id":"2101","subtype":"Figure","type":"Plot"},{"attributes":{"axis_label":"price_change_percentage_24h","coordinates":null,"formatter":{"id":"2132"},"group":null,"major_label_policy":{"id":"2133"},"ticker":{"id":"2111"}},"id":"2110","type":"LinearAxis"},{"attributes":{},"id":"2111","type":"BasicTicker"},{"attributes":{"end":2.997678656273595,"reset_end":2.997678656273595,"reset_start":-2.107454305728652,"start":-2.107454305728652,"tags":[[["price_change_percentage_7d","price_change_percentage_7d",null]]]},"id":"2099","type":"Range1d"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"2140","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02149","sizing_mode":"stretch_width"},"id":"2097","type":"Spacer"},{"attributes":{},"id":"2108","type":"LinearScale"},{"attributes":{"children":[{"id":"2097"},{"id":"2101"},{"id":"2168"}],"margin":[0,0,0,0],"name":"Row02145","tags":["embedded"]},"id":"2096","type":"Row"},{"attributes":{"axis":{"id":"2110"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2113","type":"Grid"},{"attributes":{},"id":"2133","type":"AllLabels"},{"attributes":{"axis_label":"price_change_percentage_7d","coordinates":null,"formatter":{"id":"2135"},"group":null,"major_label_policy":{"id":"2136"},"ticker":{"id":"2115"}},"id":"2114","type":"LinearAxis"},{"attributes":{"axis":{"id":"2114"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2117","type":"Grid"},{"attributes":{},"id":"2115","type":"BasicTicker"},{"attributes":{},"id":"2120","type":"WheelZoomTool"},{"attributes":{},"id":"2118","type":"SaveTool"},{"attributes":{"overlay":{"id":"2123"}},"id":"2121","type":"BoxZoomTool"},{"attributes":{},"id":"2122","type":"ResetTool"},{"attributes":{},"id":"2132","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2123","type":"BoxAnnotation"}],"root_ids":["2096"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"7efd9d04-1772-4ed2-8989-aa065866e2bd","root_ids":["2096"],"roots":{"2096":"00217166-eb1f-4013-a427-86f3961e5566"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Optimize Clusters with Principal Component Analysis

In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

1. Create a PCA model instance and set `n_components=3`.

2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 

3. Retrieve the explained variance to determine how much information can be attributed to each principal component.

4. Answer the following question: What is the total explained variance of the three principal components?

5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.


```python
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)
```


```python
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_pca_data = pca.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
market_pca_data
```




    array([[-0.60066733,  0.84276006,  0.46159457],
           [-0.45826071,  0.45846566,  0.95287678],
           [-0.43306981, -0.16812638, -0.64175193],
           [-0.47183495, -0.22266008, -0.47905316],
           [-1.15779997,  2.04120919,  1.85971527],
           [-0.51653377,  1.38837748,  0.80407131],
           [-0.45071134,  0.51769912,  2.84614316],
           [-0.34559977,  0.72943939,  1.47801284],
           [-0.64946792,  0.43216514,  0.60030286],
           [-0.75901394, -0.20119979, -0.21765292],
           [-0.24819846, -1.37625159, -1.46202571],
           [-0.43840762, -0.17533654, -0.6633884 ],
           [-0.69342533, -0.47381462, -0.52759693],
           [ 0.06049915,  2.90940385,  1.49857131],
           [-0.39335243, -0.10819197, -0.01275608],
           [-0.79617564, -0.49440875,  1.08281169],
           [ 0.06407452, -1.26982514, -1.09882928],
           [-0.48901506, -0.73271912, -0.06254323],
           [-0.3062723 ,  0.70341515,  1.71422359],
           [-0.51352775, -0.14280239, -0.65656583],
           [-0.36212044, -0.98691441, -0.72875232],
           [-0.60426463,  0.82739764,  0.43931594],
           [-0.4132956 , -0.67411527, -1.07662834],
           [-0.40748304, -0.21250655, -0.35142563],
           [ 0.60897382,  0.56353212, -1.14874159],
           [-0.45021114, -0.15101945, -0.64740061],
           [-0.76466522, -0.51788554,  0.20499029],
           [-0.55631468, -1.93820906, -1.26177589],
           [-0.42514677,  0.49297617,  1.05804837],
           [ 2.67686761, -0.0139541 , -1.96520722],
           [-0.61392275, -0.4793368 ,  0.33956513],
           [-0.57992398, -0.35633377, -0.11494202],
           [ 8.08901821, -3.89689054,  2.30138208],
           [-0.38904526,  0.16504063,  0.3794137 ],
           [ 0.86576183, -2.26188239,  0.27558289],
           [ 0.11167508,  0.42831576, -1.20539797],
           [ 4.7923954 ,  6.76767868, -1.98698545],
           [-0.63235492, -2.10811713, -0.65222738],
           [-0.59314216,  0.02148496,  0.20991142],
           [-0.4581305 , -0.13573403, -0.63528357],
           [-0.29791045, -0.1911256 , -0.90960173]])




```python
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_
```




    array([0.3719856 , 0.34700813, 0.17603793])



#### Answer the following question: What is the total explained variance of the three principal components?

**Question:** What is the total explained variance of the three principal components?

**Answer:** 


```python
# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data
df_market_data_pca = pd.DataFrame(
    market_pca_data,
    columns=["PC1", "PC2", "PC3"])

# Copy the crypto names from the original data
df_market_data_pca["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_pca = df_market_data_pca.set_index("coin_id")

# Display sample data
df_market_data_pca.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>-0.600667</td>
      <td>0.842760</td>
      <td>0.461595</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>-0.458261</td>
      <td>0.458466</td>
      <td>0.952877</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.433070</td>
      <td>-0.168126</td>
      <td>-0.641752</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.471835</td>
      <td>-0.222660</td>
      <td>-0.479053</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>-1.157800</td>
      <td>2.041209</td>
      <td>1.859715</td>
    </tr>
  </tbody>
</table>
</div>



---

### Find the Best Value for k Using the PCA Data

In this section, you will use the elbow method to find the best value for `k` using the PCA data.

1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?


```python
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
```


```python
# Create an empy list to store the inertia values
inertia = []
```


```python
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_pca)
    inertia.append(model.inertia_)
```


```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)
```


```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow.hvplot.line(x="k" , y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot
```






<div id='2220'>
  <div class="bk-root" id="4c7475d6-ed89-4198-b16a-fc8def405b3b" data-root-id="2220"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"13430b5c-e879-4848-8761-ead1aa3529f7":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02435","sizing_mode":"stretch_width"},"id":"2282","type":"Spacer"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02434","sizing_mode":"stretch_width"},"id":"2221","type":"Spacer"},{"attributes":{},"id":"2272","type":"AllLabels"},{"attributes":{"below":[{"id":"2234"}],"center":[{"id":"2237"},{"id":"2241"}],"height":300,"left":[{"id":"2238"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2261"}],"sizing_mode":"fixed","title":{"id":"2226"},"toolbar":{"id":"2248"},"width":700,"x_range":{"id":"2222"},"x_scale":{"id":"2230"},"y_range":{"id":"2223"},"y_scale":{"id":"2232"}},"id":"2225","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"2267","type":"AllLabels"},{"attributes":{"callback":null,"renderers":[{"id":"2261"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"2224","type":"HoverTool"},{"attributes":{"source":{"id":"2255"}},"id":"2262","type":"CDSView"},{"attributes":{},"id":"2230","type":"LinearScale"},{"attributes":{"coordinates":null,"data_source":{"id":"2255"},"glyph":{"id":"2258"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2260"},"nonselection_glyph":{"id":"2259"},"selection_glyph":{"id":"2263"},"view":{"id":"2262"}},"id":"2261","type":"GlyphRenderer"},{"attributes":{},"id":"2232","type":"LinearScale"},{"attributes":{},"id":"2243","type":"PanTool"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"2226","type":"Title"},{"attributes":{"axis":{"id":"2234"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2237","type":"Grid"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"2266"},"group":null,"major_label_policy":{"id":"2267"},"ticker":{"id":"2264"}},"id":"2234","type":"LinearAxis"},{"attributes":{},"id":"2271","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2259","type":"Line"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"2271"},"group":null,"major_label_policy":{"id":"2272"},"ticker":{"id":"2239"}},"id":"2238","type":"LinearAxis"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2260","type":"Line"},{"attributes":{"axis":{"id":"2238"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2241","type":"Grid"},{"attributes":{},"id":"2239","type":"BasicTicker"},{"attributes":{},"id":"2266","type":"BasicTickFormatter"},{"attributes":{},"id":"2244","type":"WheelZoomTool"},{"attributes":{"end":288.2,"reset_end":288.2,"reset_start":285.8,"start":285.8,"tags":[[["inertia","inertia",null]]]},"id":"2223","type":"Range1d"},{"attributes":{},"id":"2242","type":"SaveTool"},{"attributes":{"overlay":{"id":"2247"}},"id":"2245","type":"BoxZoomTool"},{"attributes":{},"id":"2246","type":"ResetTool"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2247","type":"BoxAnnotation"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2258","type":"Line"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2263","type":"Line"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"2222","type":"Range1d"},{"attributes":{},"id":"2256","type":"Selection"},{"attributes":{"tools":[{"id":"2224"},{"id":"2242"},{"id":"2243"},{"id":"2244"},{"id":"2245"},{"id":"2246"}]},"id":"2248","type":"Toolbar"},{"attributes":{},"id":"2279","type":"UnionRenderers"},{"attributes":{"data":{"inertia":{"__ndarray__":"AAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"2256"},"selection_policy":{"id":"2279"}},"id":"2255","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"2221"},{"id":"2225"},{"id":"2282"}],"margin":[0,0,0,0],"name":"Row02430","tags":["embedded"]},"id":"2220","type":"Row"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"2264","type":"FixedTicker"}],"root_ids":["2220"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"13430b5c-e879-4848-8761-ead1aa3529f7","root_ids":["2220"],"roots":{"2220":"4c7475d6-ed89-4198-b16a-fc8def405b3b"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
* **Question:** What is the best value for `k` when using the PCA data?

  * **Answer:** # YOUR ANSWER HERE!


* **Question:** Does it differ from the best k value found using the original data?

  * **Answer:** # YOUR ANSWER HERE!

---

### Cluster Cryptocurrencies with K-means Using the PCA Data

In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the PCA data.

3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.

4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.


```python
# Initialize the K-Means model using the best value for k
model_pca = KMeans(n_clusters=4)
```


```python
# Fit the K-Means model using the PCA data
model_pca.fit(df_market_data_pca)
```




    KMeans(n_clusters=4)




```python
# Predict the clusters to group the cryptocurrencies using the PCA data
crypto_clusters_pca = model_pca.predict(df_market_data_pca)

# View the resulting array of cluster values.
print(crypto_clusters_pca)
```

    [3 3 0 0 3 3 3 3 3 0 0 0 0 3 0 3 0 0 3 0 0 3 0 0 0 0 0 0 3 0 0 0 1 3 0 0 2
     0 0 0 0]



```python
# Create a copy of the DataFrame with the PCA data
df_market_data_pca_predictions = df_market_data_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_market_data_pca_predictions["crypto_cluster"] = crypto_clusters_pca

# Display sample data
df_market_data_pca_predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>crypto_cluster</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>-0.600667</td>
      <td>0.842760</td>
      <td>0.461595</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>-0.458261</td>
      <td>0.458466</td>
      <td>0.952877</td>
      <td>3</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.433070</td>
      <td>-0.168126</td>
      <td>-0.641752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.471835</td>
      <td>-0.222660</td>
      <td>-0.479053</td>
      <td>0</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>-1.157800</td>
      <td>2.041209</td>
      <td>1.859715</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
clusters_plot_pca = df_market_data_pca_predictions.hvplot.scatter(
    x="PC1",
    y="PC2",
    by="crypto_cluster",
    hover_cols=["coin_id"],
    marker=["hex", "square", "cross", "inverted_triangle"],
    title="Cryptocurrencies Clusters Using PCA Data",  
)    
clusters_plot_pca 
```






<div id='2329'>
  <div class="bk-root" id="3a963272-daaa-4dd8-911e-5756c2f66a21" data-root-id="2329"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"ddfa6a4d-a764-4c11-8d15-66103ad76df2":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"source":{"id":"2416"}},"id":"2423","type":"CDSView"},{"attributes":{"data":{"PC1":{"__ndarray__":"N/XQrqo447+40e2+JFTdvz0Tj0NZhvK/7NIJ1HGH4L/CWkdgdNjcv9b9x3tOHta/h21M83DI5L/guE+KvvmuP6YxxVNFeum/o86LH/eZ07/B3W/HIlbjvzzz5tCaNdu/uEmjFh7m2L8=","dtype":"float64","order":"little","shape":[13]},"PC2":{"__ndarray__":"ygJY8+P36j9/P81egFfdP6tEN3tlVABAdqRxTss29j+9ZPC8/ZDgP4bE/UiRV+c/ey0795eo2z+U9tOGdUYHQEqknptkpN+/akNue2CC5j9mCxibCnrqP+V0su7rjN8/W0ntLg0gxT8=","dtype":"float64","order":"little","shape":[13]},"coin_id":["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"],"crypto_cluster":[3,3,3,3,3,3,3,3,3,3,3,3,3]},"selected":{"id":"2441"},"selection_policy":{"id":"2461"}},"id":"2440","type":"ColumnDataSource"},{"attributes":{},"id":"2371","type":"BasicTickFormatter"},{"attributes":{},"id":"2372","type":"AllLabels"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"2400"}]},"id":"2414","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.2},"line_color":{"value":"#6d904f"},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2445","type":"Scatter"},{"attributes":{"end":8.485310422788723,"reset_end":8.485310422788723,"reset_start":-1.5540921804637515,"start":-1.5540921804637515,"tags":[[["PC1","PC1",null]]]},"id":"2331","type":"Range1d"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"2379"}]},"id":"2392","type":"LegendItem"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#fc4f30"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#fc4f30"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2415","type":"Scatter"},{"attributes":{"callback":null,"renderers":[{"id":"2379"},{"id":"2400"},{"id":"2422"},{"id":"2446"}],"tags":["hv_created"],"tooltips":[["crypto_cluster","@{crypto_cluster}"],["PC1","@{PC1}"],["PC2","@{PC2}"],["coin_id","@{coin_id}"]]},"id":"2333","type":"HoverTool"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"2392"},{"id":"2414"},{"id":"2438"},{"id":"2464"}],"location":[0,0],"title":"crypto_cluster"},"id":"2391","type":"Legend"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#6d904f"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#6d904f"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2465","type":"Scatter"},{"attributes":{"end":7.834135596337129,"reset_end":7.834135596337129,"reset_start":-4.963347456561397,"start":-4.963347456561397,"tags":[[["PC2","PC2",null]]]},"id":"2332","type":"Range1d"},{"attributes":{},"id":"2417","type":"Selection"},{"attributes":{"source":{"id":"2373"}},"id":"2380","type":"CDSView"},{"attributes":{"coordinates":null,"data_source":{"id":"2373"},"glyph":{"id":"2376"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2378"},"nonselection_glyph":{"id":"2377"},"selection_glyph":{"id":"2393"},"view":{"id":"2380"}},"id":"2379","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#6d904f"},"hatch_color":{"value":"#6d904f"},"line_color":{"value":"#6d904f"},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2443","type":"Scatter"},{"attributes":{},"id":"2342","type":"LinearScale"},{"attributes":{"coordinates":null,"data_source":{"id":"2394"},"glyph":{"id":"2397"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2399"},"nonselection_glyph":{"id":"2398"},"selection_glyph":{"id":"2415"},"view":{"id":"2401"}},"id":"2400","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2378","type":"Scatter"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2393","type":"Scatter"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#e5ae38"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#e5ae38"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2439","type":"Scatter"},{"attributes":{},"id":"2411","type":"UnionRenderers"},{"attributes":{},"id":"2395","type":"Selection"},{"attributes":{},"id":"2344","type":"LinearScale"},{"attributes":{},"id":"2461","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"line_color":{"value":"#fc4f30"},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2399","type":"Scatter"},{"attributes":{"below":[{"id":"2346"}],"center":[{"id":"2349"},{"id":"2353"}],"height":300,"left":[{"id":"2350"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2379"},{"id":"2400"},{"id":"2422"},{"id":"2446"}],"right":[{"id":"2391"}],"sizing_mode":"fixed","title":{"id":"2338"},"toolbar":{"id":"2360"},"width":700,"x_range":{"id":"2331"},"x_scale":{"id":"2342"},"y_range":{"id":"2332"},"y_scale":{"id":"2344"}},"id":"2337","subtype":"Figure","type":"Plot"},{"attributes":{"coordinates":null,"group":null,"text":"Cryptocurrencies Clusters Using PCA Data","text_color":"black","text_font_size":"12pt"},"id":"2338","type":"Title"},{"attributes":{},"id":"2441","type":"Selection"},{"attributes":{"axis_label":"PC1","coordinates":null,"formatter":{"id":"2368"},"group":null,"major_label_policy":{"id":"2369"},"ticker":{"id":"2347"}},"id":"2346","type":"LinearAxis"},{"attributes":{},"id":"2355","type":"PanTool"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"2422"}]},"id":"2438","type":"LegendItem"},{"attributes":{"axis":{"id":"2346"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2349","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"line_color":{"value":"#fc4f30"},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2398","type":"Scatter"},{"attributes":{"source":{"id":"2394"}},"id":"2401","type":"CDSView"},{"attributes":{},"id":"2347","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.1},"line_color":{"value":"#6d904f"},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2444","type":"Scatter"},{"attributes":{},"id":"2369","type":"AllLabels"},{"attributes":{"axis_label":"PC2","coordinates":null,"formatter":{"id":"2371"},"group":null,"major_label_policy":{"id":"2372"},"ticker":{"id":"2351"}},"id":"2350","type":"LinearAxis"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02688","sizing_mode":"stretch_width"},"id":"2482","type":"Spacer"},{"attributes":{"axis":{"id":"2350"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2353","type":"Grid"},{"attributes":{},"id":"2351","type":"BasicTicker"},{"attributes":{},"id":"2356","type":"WheelZoomTool"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02687","sizing_mode":"stretch_width"},"id":"2330","type":"Spacer"},{"attributes":{"coordinates":null,"data_source":{"id":"2416"},"glyph":{"id":"2419"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2421"},"nonselection_glyph":{"id":"2420"},"selection_glyph":{"id":"2439"},"view":{"id":"2423"}},"id":"2422","type":"GlyphRenderer"},{"attributes":{},"id":"2354","type":"SaveTool"},{"attributes":{"source":{"id":"2440"}},"id":"2447","type":"CDSView"},{"attributes":{"overlay":{"id":"2359"}},"id":"2357","type":"BoxZoomTool"},{"attributes":{"coordinates":null,"data_source":{"id":"2440"},"glyph":{"id":"2443"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2445"},"nonselection_glyph":{"id":"2444"},"selection_glyph":{"id":"2465"},"view":{"id":"2447"}},"id":"2446","type":"GlyphRenderer"},{"attributes":{"data":{"PC1":{"__ndarray__":"UpZ8amq3278kYXw2izLev+Lya5vXSei/BPxzoPfEz7+Qh/TX3g7cv5wZGVOKMOa/g0wjp68s2b+6A7sJMGewP9+yldMFTN+/E+kqwdFu4L8ZF4kx+yzXv8YSO2dvc9q/1hEoujMU2r+LZdmqtnzjP/NN1mNC0Ny/mdi9MiN46L8qsu9uVM3hv+VGFZE5agVAqSC8UEGl47/DvuS6vI7ivzEKKydStOs/0pK6/7yWvD+wXMBgQDzkv5UbmkYF++K/Gr1JmQJS3b8X4q/49hDTvw==","dtype":"float64","order":"little","shape":[26]},"PC2":{"__ndarray__":"Z9F7TyqFxb9wj1UaIIDMv1DhzCbqwMm/xToUYyAF9r+JZsaEbXHGvxbJIJL6Ut6/DSu0IHiyu79wgFApNFH0v/egJl5vcue/k5KaSllHwr+tiEuGzZTvv2D5wy5akuW/it3jHmozy7+eF2SFdAjiPzMuIf2aVMO/H+uFsYSS4L9QbbiA5wL/vwyKJoz3k4y/c48PRXSt3r/z3rYoLM7Wv2z+1spVGALA+SEohoZp2z89qFKDbN0AwCsR7msnAJY/HvWtmbtfwb8FSPDBzXbIvw==","dtype":"float64","order":"little","shape":[26]},"coin_id":["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"],"crypto_cluster":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"selected":{"id":"2374"},"selection_policy":{"id":"2388"}},"id":"2373","type":"ColumnDataSource"},{"attributes":{},"id":"2358","type":"ResetTool"},{"attributes":{},"id":"2388","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2359","type":"BoxAnnotation"},{"attributes":{},"id":"2374","type":"Selection"},{"attributes":{"data":{"PC1":{"__ndarray__":"YBQ7s2krE0A=","dtype":"float64","order":"little","shape":[1]},"PC2":{"__ndarray__":"FlrQWxoSG0A=","dtype":"float64","order":"little","shape":[1]},"coin_id":["celsius-degree-token"],"crypto_cluster":[2]},"selected":{"id":"2417"},"selection_policy":{"id":"2435"}},"id":"2416","type":"ColumnDataSource"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"2446"}]},"id":"2464","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2377","type":"Scatter"},{"attributes":{"fill_color":{"value":"#fc4f30"},"hatch_color":{"value":"#fc4f30"},"line_color":{"value":"#fc4f30"},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2397","type":"Scatter"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2376","type":"Scatter"},{"attributes":{"tools":[{"id":"2333"},{"id":"2354"},{"id":"2355"},{"id":"2356"},{"id":"2357"},{"id":"2358"}]},"id":"2360","type":"Toolbar"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"line_color":{"value":"#e5ae38"},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2421","type":"Scatter"},{"attributes":{"fill_color":{"value":"#e5ae38"},"hatch_color":{"value":"#e5ae38"},"line_color":{"value":"#e5ae38"},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2419","type":"Scatter"},{"attributes":{},"id":"2368","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"line_color":{"value":"#e5ae38"},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2420","type":"Scatter"},{"attributes":{"data":{"PC1":{"__ndarray__":"RlWiy5MtIEA=","dtype":"float64","order":"little","shape":[1]},"PC2":{"__ndarray__":"YQ3w8dQsD8A=","dtype":"float64","order":"little","shape":[1]},"coin_id":["ethlend"],"crypto_cluster":[1]},"selected":{"id":"2395"},"selection_policy":{"id":"2411"}},"id":"2394","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"2330"},{"id":"2337"},{"id":"2482"}],"margin":[0,0,0,0],"name":"Row02683","tags":["embedded"]},"id":"2329","type":"Row"},{"attributes":{},"id":"2435","type":"UnionRenderers"}],"root_ids":["2329"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"ddfa6a4d-a764-4c11-8d15-66103ad76df2","root_ids":["2329"],"roots":{"2329":"3a963272-daaa-4dd8-911e-5756c2f66a21"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Visualize and Compare the Results

In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.

2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.

3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

> **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).


```python
# Composite plot to contrast the Elbow curves
elbow_plot + elbow_plot_pca
```






<div id='2564'>
  <div class="bk-root" id="e69eba7b-f8e1-4d86-81f4-dbc0397b2c97" data-root-id="2564"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"d2bfede4-4de5-4eb1-b28c-429e08e2cee6":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"axis":{"id":"2631"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2634","type":"Grid"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2640","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2603","type":"Line"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2656","type":"Line"},{"attributes":{},"id":"2682","type":"UnionRenderers"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02787","sizing_mode":"stretch_width"},"id":"2708","type":"Spacer"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2651","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2652","type":"Line"},{"attributes":{},"id":"2639","type":"ResetTool"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2607","type":"Line"},{"attributes":{},"id":"2600","type":"Selection"},{"attributes":{},"id":"2637","type":"WheelZoomTool"},{"attributes":{},"id":"2610","type":"BasicTickFormatter"},{"attributes":{},"id":"2672","type":"UnionRenderers"},{"attributes":{"tools":[{"id":"2617"},{"id":"2635"},{"id":"2636"},{"id":"2637"},{"id":"2638"},{"id":"2639"}]},"id":"2641","type":"Toolbar"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2604","type":"Line"},{"attributes":{"coordinates":null,"data_source":{"id":"2599"},"glyph":{"id":"2602"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2604"},"nonselection_glyph":{"id":"2603"},"selection_glyph":{"id":"2607"},"view":{"id":"2606"}},"id":"2605","type":"GlyphRenderer"},{"attributes":{"source":{"id":"2599"}},"id":"2606","type":"CDSView"},{"attributes":{},"id":"2659","type":"BasicTickFormatter"},{"attributes":{},"id":"2632","type":"BasicTicker"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2653","type":"Line"},{"attributes":{"end":288.2,"reset_end":288.2,"reset_start":285.8,"start":285.8,"tags":[[["inertia","inertia",null]]]},"id":"2567","type":"Range1d"},{"attributes":{"coordinates":null,"data_source":{"id":"2648"},"glyph":{"id":"2651"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2653"},"nonselection_glyph":{"id":"2652"},"selection_glyph":{"id":"2656"},"view":{"id":"2655"}},"id":"2654","type":"GlyphRenderer"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"2608","type":"FixedTicker"},{"attributes":{"source":{"id":"2648"}},"id":"2655","type":"CDSView"},{"attributes":{"below":[{"id":"2627"}],"center":[{"id":"2630"},{"id":"2634"}],"height":300,"left":[{"id":"2631"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2654"}],"sizing_mode":"fixed","title":{"id":"2619"},"toolbar":{"id":"2641"},"toolbar_location":null,"width":700,"x_range":{"id":"2566"},"x_scale":{"id":"2623"},"y_range":{"id":"2567"},"y_scale":{"id":"2625"}},"id":"2618","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"inertia":{"__ndarray__":"AAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"2600"},"selection_policy":{"id":"2672"}},"id":"2599","type":"ColumnDataSource"},{"attributes":{"data":{"inertia":{"__ndarray__":"AAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"2649"},"selection_policy":{"id":"2682"}},"id":"2648","type":"ColumnDataSource"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"2657","type":"FixedTicker"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"2566","type":"Range1d"},{"attributes":{"children":[{"id":"2688"},{"id":"2686"}]},"id":"2689","type":"Column"},{"attributes":{},"id":"2574","type":"LinearScale"},{"attributes":{},"id":"2616","type":"AllLabels"},{"attributes":{},"id":"2611","type":"AllLabels"},{"attributes":{},"id":"2625","type":"LinearScale"},{"attributes":{"axis":{"id":"2627"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2630","type":"Grid"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2602","type":"Line"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve Using PCA Data","text_color":"black","text_font_size":"12pt"},"id":"2619","type":"Title"},{"attributes":{"below":[{"id":"2578"}],"center":[{"id":"2581"},{"id":"2585"}],"height":300,"left":[{"id":"2582"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2605"}],"sizing_mode":"fixed","title":{"id":"2570"},"toolbar":{"id":"2592"},"toolbar_location":null,"width":700,"x_range":{"id":"2566"},"x_scale":{"id":"2574"},"y_range":{"id":"2567"},"y_scale":{"id":"2576"}},"id":"2569","subtype":"Figure","type":"Plot"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"2664"},"group":null,"major_label_policy":{"id":"2665"},"ticker":{"id":"2632"}},"id":"2631","type":"LinearAxis"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"2659"},"group":null,"major_label_policy":{"id":"2660"},"ticker":{"id":"2657"}},"id":"2627","type":"LinearAxis"},{"attributes":{"toolbars":[{"id":"2592"},{"id":"2641"}],"tools":[{"id":"2568"},{"id":"2586"},{"id":"2587"},{"id":"2588"},{"id":"2589"},{"id":"2590"},{"id":"2617"},{"id":"2635"},{"id":"2636"},{"id":"2637"},{"id":"2638"},{"id":"2639"}]},"id":"2687","type":"ProxyToolbar"},{"attributes":{"callback":null,"renderers":[{"id":"2605"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"2568","type":"HoverTool"},{"attributes":{},"id":"2576","type":"LinearScale"},{"attributes":{},"id":"2587","type":"PanTool"},{"attributes":{},"id":"2635","type":"SaveTool"},{"attributes":{"children":[{"id":"2565"},{"id":"2689"},{"id":"2708"}],"margin":[0,0,0,0],"name":"Row02782","tags":["embedded"]},"id":"2564","type":"Row"},{"attributes":{},"id":"2636","type":"PanTool"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"2570","type":"Title"},{"attributes":{"axis":{"id":"2578"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2581","type":"Grid"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"2610"},"group":null,"major_label_policy":{"id":"2611"},"ticker":{"id":"2608"}},"id":"2578","type":"LinearAxis"},{"attributes":{"children":[[{"id":"2569"},0,0],[{"id":"2618"},0,1]]},"id":"2686","type":"GridBox"},{"attributes":{"callback":null,"renderers":[{"id":"2654"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"2617","type":"HoverTool"},{"attributes":{"toolbar":{"id":"2687"},"toolbar_location":"above"},"id":"2688","type":"ToolbarBox"},{"attributes":{},"id":"2615","type":"BasicTickFormatter"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"2615"},"group":null,"major_label_policy":{"id":"2616"},"ticker":{"id":"2583"}},"id":"2582","type":"LinearAxis"},{"attributes":{"overlay":{"id":"2640"}},"id":"2638","type":"BoxZoomTool"},{"attributes":{"axis":{"id":"2582"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2585","type":"Grid"},{"attributes":{},"id":"2665","type":"AllLabels"},{"attributes":{},"id":"2583","type":"BasicTicker"},{"attributes":{},"id":"2588","type":"WheelZoomTool"},{"attributes":{},"id":"2586","type":"SaveTool"},{"attributes":{},"id":"2664","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2591","type":"BoxAnnotation"},{"attributes":{},"id":"2660","type":"AllLabels"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02786","sizing_mode":"stretch_width"},"id":"2565","type":"Spacer"},{"attributes":{"overlay":{"id":"2591"}},"id":"2589","type":"BoxZoomTool"},{"attributes":{},"id":"2649","type":"Selection"},{"attributes":{},"id":"2590","type":"ResetTool"},{"attributes":{},"id":"2623","type":"LinearScale"},{"attributes":{"tools":[{"id":"2568"},{"id":"2586"},{"id":"2587"},{"id":"2588"},{"id":"2589"},{"id":"2590"}]},"id":"2592","type":"Toolbar"}],"root_ids":["2564"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"d2bfede4-4de5-4eb1-b28c-429e08e2cee6","root_ids":["2564"],"roots":{"2564":"e69eba7b-f8e1-4d86-81f4-dbc0397b2c97"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
# Compoosite plot to contrast the clusters
elbow_plot + clusters_plot_pca
```






<div id='2800'>
  <div class="bk-root" id="5cecd32b-cc61-43ef-8f16-351029813045" data-root-id="2800"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"c9986cb4-4a89-44ab-a022-d79923421d4b":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"NotificationAreaBase1","overrides":[],"properties":[{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"}]},{"extends":null,"module":null,"name":"NotificationArea1","overrides":[],"properties":[{"default":[],"kind":null,"name":"notifications"},{"default":"bottom-right","kind":null,"name":"position"},{"default":0,"kind":null,"name":"_clear"},{"default":[{"background":"#ffc107","icon":{"className":"fas fa-exclamation-triangle","color":"white","tagName":"i"},"type":"warning"},{"background":"#007bff","icon":{"className":"fas fa-info-circle","color":"white","tagName":"i"},"type":"info"}],"kind":null,"name":"types"}]},{"extends":null,"module":null,"name":"Notification","overrides":[],"properties":[{"default":null,"kind":null,"name":"background"},{"default":3000,"kind":null,"name":"duration"},{"default":null,"kind":null,"name":"icon"},{"default":"","kind":null,"name":"message"},{"default":null,"kind":null,"name":"notification_type"},{"default":false,"kind":null,"name":"_destroyed"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{},"id":"2983","type":"UnionRenderers"},{"attributes":{},"id":"2894","type":"AllLabels"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"2944"}]},"id":"2960","type":"LegendItem"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"2968"}]},"id":"2986","type":"LegendItem"},{"attributes":{},"id":"2812","type":"LinearScale"},{"attributes":{"toolbar":{"id":"3016"},"toolbar_location":"above"},"id":"3017","type":"ToolbarBox"},{"attributes":{"coordinates":null,"data_source":{"id":"2962"},"glyph":{"id":"2965"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2967"},"nonselection_glyph":{"id":"2966"},"selection_glyph":{"id":"2987"},"view":{"id":"2969"}},"id":"2968","type":"GlyphRenderer"},{"attributes":{},"id":"2823","type":"PanTool"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.2},"line_color":{"value":"#6d904f"},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2967","type":"Scatter"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"2806","type":"Title"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2899","type":"Scatter"},{"attributes":{"source":{"id":"2962"}},"id":"2969","type":"CDSView"},{"attributes":{"axis":{"id":"2814"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2817","type":"Grid"},{"attributes":{},"id":"2994","type":"UnionRenderers"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"2846"},"group":null,"major_label_policy":{"id":"2847"},"ticker":{"id":"2844"}},"id":"2814","type":"LinearAxis"},{"attributes":{},"id":"2896","type":"Selection"},{"attributes":{"source":{"id":"2895"}},"id":"2902","type":"CDSView"},{"attributes":{},"id":"2851","type":"BasicTickFormatter"},{"attributes":{"end":7.834135596337129,"reset_end":7.834135596337129,"reset_start":-4.963347456561397,"start":-4.963347456561397,"tags":[[["PC2","PC2",null]]]},"id":"2854","type":"Range1d"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2900","type":"Scatter"},{"attributes":{},"id":"2869","type":"BasicTicker"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"2851"},"group":null,"major_label_policy":{"id":"2852"},"ticker":{"id":"2819"}},"id":"2818","type":"LinearAxis"},{"attributes":{"coordinates":null,"data_source":{"id":"2895"},"glyph":{"id":"2898"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2900"},"nonselection_glyph":{"id":"2899"},"selection_glyph":{"id":"2915"},"view":{"id":"2902"}},"id":"2901","type":"GlyphRenderer"},{"attributes":{"coordinates":null,"data_source":{"id":"2916"},"glyph":{"id":"2919"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2921"},"nonselection_glyph":{"id":"2920"},"selection_glyph":{"id":"2937"},"view":{"id":"2923"}},"id":"2922","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.1},"line_color":{"value":"#6d904f"},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2966","type":"Scatter"},{"attributes":{"axis":{"id":"2818"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2821","type":"Grid"},{"attributes":{"children":[{"id":"2801"},{"id":"3018"},{"id":"3044"}],"margin":[0,0,0,0],"name":"Row02860","tags":["embedded"]},"id":"2800","type":"Row"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"2901"}]},"id":"2914","type":"LegendItem"},{"attributes":{},"id":"2819","type":"BasicTicker"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2915","type":"Scatter"},{"attributes":{},"id":"2824","type":"WheelZoomTool"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"2914"},{"id":"2936"},{"id":"2960"},{"id":"2986"}],"location":[0,0],"title":"crypto_cluster"},"id":"2913","type":"Legend"},{"attributes":{},"id":"2866","type":"LinearScale"},{"attributes":{"fill_color":{"value":"#e5ae38"},"hatch_color":{"value":"#e5ae38"},"line_color":{"value":"#e5ae38"},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2941","type":"Scatter"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"2922"}]},"id":"2936","type":"LegendItem"},{"attributes":{},"id":"2822","type":"SaveTool"},{"attributes":{"axis":{"id":"2872"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2875","type":"Grid"},{"attributes":{"coordinates":null,"group":null,"text":"Cryptocurrencies Clusters Using PCA Data","text_color":"black","text_font_size":"12pt"},"id":"2860","type":"Title"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"line_color":{"value":"#e5ae38"},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2942","type":"Scatter"},{"attributes":{"overlay":{"id":"2827"}},"id":"2825","type":"BoxZoomTool"},{"attributes":{},"id":"2877","type":"PanTool"},{"attributes":{},"id":"2826","type":"ResetTool"},{"attributes":{},"id":"2847","type":"AllLabels"},{"attributes":{"source":{"id":"2938"}},"id":"2945","type":"CDSView"},{"attributes":{},"id":"2917","type":"Selection"},{"attributes":{"end":8.485310422788723,"reset_end":8.485310422788723,"reset_start":-1.5540921804637515,"start":-1.5540921804637515,"tags":[[["PC1","PC1",null]]]},"id":"2853","type":"Range1d"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"line_color":{"value":"#e5ae38"},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2943","type":"Scatter"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2827","type":"BoxAnnotation"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"line_color":{"value":"#fc4f30"},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2921","type":"Scatter"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2843","type":"Line"},{"attributes":{"data":{"PC1":{"__ndarray__":"RlWiy5MtIEA=","dtype":"float64","order":"little","shape":[1]},"PC2":{"__ndarray__":"YQ3w8dQsD8A=","dtype":"float64","order":"little","shape":[1]},"coin_id":["ethlend"],"crypto_cluster":[1]},"selected":{"id":"2917"},"selection_policy":{"id":"2933"}},"id":"2916","type":"ColumnDataSource"},{"attributes":{"coordinates":null,"data_source":{"id":"2938"},"glyph":{"id":"2941"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2943"},"nonselection_glyph":{"id":"2942"},"selection_glyph":{"id":"2961"},"view":{"id":"2945"}},"id":"2944","type":"GlyphRenderer"},{"attributes":{"below":[{"id":"2868"}],"center":[{"id":"2871"},{"id":"2875"}],"height":300,"left":[{"id":"2872"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2901"},{"id":"2922"},{"id":"2944"},{"id":"2968"}],"right":[{"id":"2913"}],"sizing_mode":"fixed","title":{"id":"2860"},"toolbar":{"id":"2882"},"toolbar_location":null,"width":700,"x_range":{"id":"2853"},"x_scale":{"id":"2864"},"y_range":{"id":"2854"},"y_scale":{"id":"2866"}},"id":"2859","subtype":"Figure","type":"Plot"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2838","type":"Line"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"marker":{"value":"hex"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2898","type":"Scatter"},{"attributes":{"callback":null,"renderers":[{"id":"2901"},{"id":"2922"},{"id":"2944"},{"id":"2968"}],"tags":["hv_created"],"tooltips":[["crypto_cluster","@{crypto_cluster}"],["PC1","@{PC1}"],["PC2","@{PC2}"],["coin_id","@{coin_id}"]]},"id":"2855","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"line_color":{"value":"#fc4f30"},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2920","type":"Scatter"},{"attributes":{"source":{"id":"2916"}},"id":"2923","type":"CDSView"},{"attributes":{"data":{"PC1":{"__ndarray__":"N/XQrqo447+40e2+JFTdvz0Tj0NZhvK/7NIJ1HGH4L/CWkdgdNjcv9b9x3tOHta/h21M83DI5L/guE+KvvmuP6YxxVNFeum/o86LH/eZ07/B3W/HIlbjvzzz5tCaNdu/uEmjFh7m2L8=","dtype":"float64","order":"little","shape":[13]},"PC2":{"__ndarray__":"ygJY8+P36j9/P81egFfdP6tEN3tlVABAdqRxTss29j+9ZPC8/ZDgP4bE/UiRV+c/ey0795eo2z+U9tOGdUYHQEqknptkpN+/akNue2CC5j9mCxibCnrqP+V0su7rjN8/W0ntLg0gxT8=","dtype":"float64","order":"little","shape":[13]},"coin_id":["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"],"crypto_cluster":[3,3,3,3,3,3,3,3,3,3,3,3,3]},"selected":{"id":"2963"},"selection_policy":{"id":"2983"}},"id":"2962","type":"ColumnDataSource"},{"attributes":{"axis_label":"PC2","coordinates":null,"formatter":{"id":"2893"},"group":null,"major_label_policy":{"id":"2894"},"ticker":{"id":"2873"}},"id":"2872","type":"LinearAxis"},{"attributes":{},"id":"2957","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2839","type":"Line"},{"attributes":{},"id":"2891","type":"AllLabels"},{"attributes":{},"id":"2873","type":"BasicTicker"},{"attributes":{},"id":"2939","type":"Selection"},{"attributes":{},"id":"2878","type":"WheelZoomTool"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#6d904f"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#6d904f"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2987","type":"Scatter"},{"attributes":{},"id":"2876","type":"SaveTool"},{"attributes":{"data":{"PC1":{"__ndarray__":"UpZ8amq3278kYXw2izLev+Lya5vXSei/BPxzoPfEz7+Qh/TX3g7cv5wZGVOKMOa/g0wjp68s2b+6A7sJMGewP9+yldMFTN+/E+kqwdFu4L8ZF4kx+yzXv8YSO2dvc9q/1hEoujMU2r+LZdmqtnzjP/NN1mNC0Ny/mdi9MiN46L8qsu9uVM3hv+VGFZE5agVAqSC8UEGl47/DvuS6vI7ivzEKKydStOs/0pK6/7yWvD+wXMBgQDzkv5UbmkYF++K/Gr1JmQJS3b8X4q/49hDTvw==","dtype":"float64","order":"little","shape":[26]},"PC2":{"__ndarray__":"Z9F7TyqFxb9wj1UaIIDMv1DhzCbqwMm/xToUYyAF9r+JZsaEbXHGvxbJIJL6Ut6/DSu0IHiyu79wgFApNFH0v/egJl5vcue/k5KaSllHwr+tiEuGzZTvv2D5wy5akuW/it3jHmozy7+eF2SFdAjiPzMuIf2aVMO/H+uFsYSS4L9QbbiA5wL/vwyKJoz3k4y/c48PRXSt3r/z3rYoLM7Wv2z+1spVGALA+SEohoZp2z89qFKDbN0AwCsR7msnAJY/HvWtmbtfwb8FSPDBzXbIvw==","dtype":"float64","order":"little","shape":[26]},"coin_id":["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"],"crypto_cluster":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"selected":{"id":"2896"},"selection_policy":{"id":"2910"}},"id":"2895","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"2881"}},"id":"2879","type":"BoxZoomTool"},{"attributes":{},"id":"2880","type":"ResetTool"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"2844","type":"FixedTicker"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2881","type":"BoxAnnotation"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#fc4f30"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#fc4f30"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2937","type":"Scatter"},{"attributes":{},"id":"2933","type":"UnionRenderers"},{"attributes":{},"id":"2852","type":"AllLabels"},{"attributes":{},"id":"2910","type":"UnionRenderers"},{"attributes":{},"id":"2864","type":"LinearScale"},{"attributes":{"children":[{"id":"3017"},{"id":"3015"}]},"id":"3018","type":"Column"},{"attributes":{"fill_color":{"value":"#fc4f30"},"hatch_color":{"value":"#fc4f30"},"line_color":{"value":"#fc4f30"},"marker":{"value":"square"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2919","type":"Scatter"},{"attributes":{"axis":{"id":"2868"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2871","type":"Grid"},{"attributes":{"tools":[{"id":"2855"},{"id":"2876"},{"id":"2877"},{"id":"2878"},{"id":"2879"},{"id":"2880"}]},"id":"2882","type":"Toolbar"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02864","sizing_mode":"stretch_width"},"id":"2801","type":"Spacer"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#e5ae38"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#e5ae38"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"cross"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2961","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02865","sizing_mode":"stretch_width"},"id":"3044","type":"Spacer"},{"attributes":{},"id":"2893","type":"BasicTickFormatter"},{"attributes":{"axis_label":"PC1","coordinates":null,"formatter":{"id":"2890"},"group":null,"major_label_policy":{"id":"2891"},"ticker":{"id":"2869"}},"id":"2868","type":"LinearAxis"},{"attributes":{},"id":"2846","type":"BasicTickFormatter"},{"attributes":{"tools":[{"id":"2804"},{"id":"2822"},{"id":"2823"},{"id":"2824"},{"id":"2825"},{"id":"2826"}]},"id":"2828","type":"Toolbar"},{"attributes":{"children":[[{"id":"2805"},0,0],[{"id":"2859"},0,1]]},"id":"3015","type":"GridBox"},{"attributes":{"toolbars":[{"id":"2828"},{"id":"2882"}],"tools":[{"id":"2804"},{"id":"2822"},{"id":"2823"},{"id":"2824"},{"id":"2825"},{"id":"2826"},{"id":"2855"},{"id":"2876"},{"id":"2877"},{"id":"2878"},{"id":"2879"},{"id":"2880"}]},"id":"3016","type":"ProxyToolbar"},{"attributes":{},"id":"2963","type":"Selection"},{"attributes":{"below":[{"id":"2814"}],"center":[{"id":"2817"},{"id":"2821"}],"height":300,"left":[{"id":"2818"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2841"}],"sizing_mode":"fixed","title":{"id":"2806"},"toolbar":{"id":"2828"},"toolbar_location":null,"width":700,"x_range":{"id":"2802"},"x_scale":{"id":"2810"},"y_range":{"id":"2803"},"y_scale":{"id":"2812"}},"id":"2805","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"2835"}},"id":"2842","type":"CDSView"},{"attributes":{"data":{"inertia":{"__ndarray__":"AAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUAAAAAAAPBxQAAAAAAA8HFAAAAAAADwcUA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"2836"},"selection_policy":{"id":"2994"}},"id":"2835","type":"ColumnDataSource"},{"attributes":{"coordinates":null,"data_source":{"id":"2835"},"glyph":{"id":"2838"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2840"},"nonselection_glyph":{"id":"2839"},"selection_glyph":{"id":"2843"},"view":{"id":"2842"}},"id":"2841","type":"GlyphRenderer"},{"attributes":{},"id":"2890","type":"BasicTickFormatter"},{"attributes":{"callback":null,"renderers":[{"id":"2841"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"2804","type":"HoverTool"},{"attributes":{},"id":"2810","type":"LinearScale"},{"attributes":{"end":288.2,"reset_end":288.2,"reset_start":285.8,"start":285.8,"tags":[[["inertia","inertia",null]]]},"id":"2803","type":"Range1d"},{"attributes":{},"id":"2836","type":"Selection"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2840","type":"Line"},{"attributes":{"fill_color":{"value":"#6d904f"},"hatch_color":{"value":"#6d904f"},"line_color":{"value":"#6d904f"},"marker":{"value":"inverted_triangle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2965","type":"Scatter"},{"attributes":{"data":{"PC1":{"__ndarray__":"YBQ7s2krE0A=","dtype":"float64","order":"little","shape":[1]},"PC2":{"__ndarray__":"FlrQWxoSG0A=","dtype":"float64","order":"little","shape":[1]},"coin_id":["celsius-degree-token"],"crypto_cluster":[2]},"selected":{"id":"2939"},"selection_policy":{"id":"2957"}},"id":"2938","type":"ColumnDataSource"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"2802","type":"Range1d"}],"root_ids":["2800"]},"title":"Bokeh Application","version":"2.4.3"}};
    var render_items = [{"docid":"c9986cb4-4a89-44ab-a022-d79923421d4b","root_ids":["2800"],"roots":{"2800":"5cecd32b-cc61-43ef-8f16-351029813045"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Answer:** # YOUR ANSWER HERE!
