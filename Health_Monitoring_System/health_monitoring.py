from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import json

app = Flask(__name__)

# MQTT setup
mqtt_client = mqtt.Client()
mqtt_client.connect('broker.hivemq.com', 1883, 60)

# MQTT on_message callback
def on_message(client, userdata, message):
    data = json.loads(message.payload.decode())
    # Process the data (e.g., save to database, analyze, etc.)
    print("Received data:", data)

mqtt_client.on_message = on_message
mqtt_client.subscribe("health/monitoring")

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json
    # Process and save data
    print("Received POST data:", data)
    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    mqtt_client.loop_start()
    app.run(port=5000)
