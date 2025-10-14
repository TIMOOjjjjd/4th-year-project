import os
import sys
import json
import geojson
import pandas as pd
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QPushButton, \
    QTextEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

# 预加载数据
df_gru = pd.read_csv("GRU_Merged_50epochs.csv")
df_gru.set_index("Zone", inplace=True)

zone_lookup_df = pd.read_csv("taxi-zone-lookup.csv")
location_to_zone = dict(zip(zone_lookup_df["LocationID"], zone_lookup_df["Zone"]))
zone_to_location = {v: k for k, v in location_to_zone.items()}  # 新增映射

edge_weight_csv = "edge_weight_matrix_with_flow.csv"
df_adj = pd.read_csv(edge_weight_csv, index_col=0)
adj_matrix = torch.tensor(df_adj.values, dtype=torch.float32)  # 稠密矩阵

# 转为稀疏图
edge_index, edge_attr = dense_to_sparse(adj_matrix)

zone_names = df_adj.index.tolist()
zone_idx_map = {zone: idx for idx, zone in enumerate(zone_names)}


# 处理地图数据
pred_csv = "final_predictions_multiscale.csv"
df_pred = pd.read_csv(pred_csv)

geojson_file = "taxi_zones.geojson"
with open(geojson_file, "r") as f:
    nyc_geojson = json.load(f)

# 创建 Zone Name 到 预测值的映射
predictions = dict(zip(df_pred["ZoneName"], df_pred["Refined_Pred"]))

# 创建 true_values_dict（从 True_Value 填充）
true_values_dict = dict(zip(df_pred["ZoneName"], df_pred["True_Value"]))

# 🚨 确保 ZoneName 和 zone_idx_map 兼容
true_values_dict = {zone_idx_map[k]: v for k, v in true_values_dict.items() if k in zone_idx_map}

# ✅ 调试信息，查看填充情况
print(f"📊 true_values_dict 里的区域数量: {len(true_values_dict)}")


min_pred, max_pred = min(predictions.values()), max(predictions.values())

# 颜色映射
from branca.colormap import LinearColormap

colormap = LinearColormap(
    colors=['blue', 'cyan', 'yellow', 'orange', 'red'],
    vmin=min_pred,
    vmax=max_pred * 0.8
)

# 更新 GeoJSON 颜色
for feature in nyc_geojson["features"]:
    zone_name = feature["properties"]["zone"]
    if zone_name in predictions:
        pred_value = predictions[zone_name]
        color = colormap(pred_value)
        feature["properties"]["Refined_Pred"] = pred_value
        feature["properties"]["fillColor"] = color
        feature["properties"]["style"] = {
            "fillColor": color,
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.8
        }
    else:
        feature["properties"]["fillColor"] = "#cccccc"
        feature["properties"]["Refined_Pred"] = "N/A"

# 生成地图 HTML
import folium

nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
folium.GeoJson(
    nyc_geojson,
    name="Taxi Zones",
    tooltip=folium.GeoJsonTooltip(fields=["zone", "Refined_Pred", "fillColor"],
                                  aliases=["Zone:", "Prediction:", "Color Code:"]),
    style_function=lambda feature: feature["properties"].get("style", {
        "fillColor": "#cccccc",  # 默认灰色
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.5
    })
).add_to(nyc_map)

nyc_map.save("nyc_taxi_prediction_map.html")


# ------------- GUI 界面 -------------
from PyQt5.QtGui import QFont

class TaxiDispatchWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NYC Taxi Dispatch System")
        self.setGeometry(100, 100, 1400, 800)

        # **创建字体**
        font = QFont("Arial", 14)  # 14px 字体
        result_font = QFont("Courier", 12)  # 代码风格的文本

        # **创建中央小部件**
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # **布局**
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # **左侧 - 调度系统**
        left_panel = QVBoxLayout()

        # **✅ 先创建 QLineEdit，再设置字体**
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("请输入目标 Zone ID (如 76)...")
        self.input_box.setFont(font)  # ✅ 设置字体大小
        left_panel.addWidget(self.input_box)

        # **✅ 先创建 QPushButton，再设置字体**
        self.continue_button = QPushButton("🔄 继续")
        self.continue_button.setFont(font)  # ✅ 设置按钮字体大小
        self.continue_button.clicked.connect(self.dispatch_taxi)
        left_panel.addWidget(self.continue_button)

        # **✅ 先创建 QTextEdit，再设置字体**
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(result_font)  # ✅ 代码风格的字体
        left_panel.addWidget(self.result_text)

        # **右侧 - 地图**
        self.browser = QWebEngineView()
        html_file = os.path.abspath("nyc_taxi_prediction_map.html")
        self.browser.setUrl(QUrl.fromLocalFile(html_file))

        # **添加到布局**
        layout.addLayout(left_panel, 2)  # 左侧占 2/5
        layout.addWidget(self.browser, 3)  # 右侧占 3/5

    def dispatch_taxi(self):
        target_zone_id = self.input_box.text().strip()
        if not target_zone_id.isdigit():
            self.result_text.setText("❌ 请输入有效的 Zone ID！")
            return

        target_zone_id = int(target_zone_id)
        if target_zone_id not in location_to_zone:
            self.result_text.setText(f"❌ Zone ID {target_zone_id} 未找到对应的名称！")
            return

        zone_name = location_to_zone[target_zone_id]
        if zone_name not in zone_idx_map:
            self.result_text.setText(f"⚠️ Zone {target_zone_id} 未找到对应索引！")
            return

        zone_index = zone_idx_map[zone_name]
        result_log = f"📍 目标区域: {zone_name} (Zone ID: {target_zone_id}) 对应索引: {zone_index}\n"

        if zone_name in df_gru.index and zone_index in true_values_dict:
            # 计算目标区域的缺口
            target_gru_pred = df_gru.loc[zone_name, "Prediction"]
            target_current = true_values_dict.get(zone_index, 0)
            target_required_cabs = target_gru_pred - target_current

            result_log += f"🚖 当前区域出租车缺口: {target_required_cabs:.2f}\n"

            # 当目标区域存在缺口时，优先调配至当前区域
            if target_required_cabs > 0:
                result_log += f"✅ 目标区域 {zone_name} 存在缺口，优先调配至当前区域。\n"
            else:
                # 当前区域充足，采用BFS逐步扩展搜索范围
                from collections import deque
                visited = {zone_index}
                queue = deque([(zone_index, 0)])  # (节点索引, 搜索层级)
                candidates = []  # 用来存储当前搜索层级中所有存在缺口的候选区域
                current_level = None

                # BFS搜索，逐层扩展
                while queue:
                    current, level = queue.popleft()
                    # 只检查非起始节点（即邻居及更远区域）
                    if level > 0:
                        neighbor_zone_name = zone_names[current]
                        if neighbor_zone_name in df_gru.index and current in true_values_dict:
                            neighbor_gru_pred = df_gru.loc[neighbor_zone_name, "Prediction"]
                            neighbor_current = true_values_dict.get(current, 0)
                            shortage = neighbor_gru_pred - neighbor_current
                            if shortage > 0:
                                # 记录当前层级候选区域
                                if current_level is None:
                                    current_level = level
                                if level == current_level:
                                    candidates.append((current, shortage))
                    # 扩展当前节点的邻居
                    neighbors = set(edge_index[1][edge_index[0] == current].cpu().numpy())
                    for nb in neighbors:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, level + 1))
                    # 如果队列中下一个节点的层级大于当前层且已找到候选区域，则退出循环
                    if queue and candidates and queue[0][1] > current_level:
                        break

                if candidates:
                    best_zone = max(candidates, key=lambda x: x[1])
                    best_zone_name = zone_names[best_zone[0]]
                    # 利用反转映射获得推荐区域的 Zone ID
                    best_zone_id = zone_to_location.get(best_zone_name, "未知")
                    result_log += f"🚗 推荐调派至邻近区域 {best_zone_name} (Zone ID: {best_zone_id}, 索引: {best_zone[0]})，缺口: {best_zone[1]:.2f}\n"
                else:
                    result_log += "⚠️ 在邻近区域及扩展搜索范围内均未找到存在缺口的区域。\n"
        else:
            result_log += "⚠️ 目标区域数据缺失。\n"

        self.result_text.setText(result_log)


if __name__ == "__main__":
    geojson_zones = {feature["properties"]["zone"] for feature in nyc_geojson["features"]}
    lookup_zones = set(zone_lookup_df["Zone"])
    pred_zones = set(df_pred["ZoneName"])

    print(f"🌍 GeoJSON 里的区域数量: {len(geojson_zones)}")
    print(f"📋 taxi-zone-lookup 里的区域数量: {len(lookup_zones)}")
    print(f"📈 预测数据里的区域数量: {len(pred_zones)}")

    # 找出不匹配的区域
    missing_in_geojson = lookup_zones - geojson_zones
    missing_in_predictions = geojson_zones - pred_zones


    app = QApplication(sys.argv)
    window = TaxiDispatchWindow()
    window.show()
    sys.exit(app.exec_())
