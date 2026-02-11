from flask import Flask, Response
from flask_socketio import SocketIO
import eventlet
eventlet.monkey_patch()
app = Flask(__name__)

last_value=None
tempstore={}
socketio=SocketIO(app,cors_allowed_origins="*",async_mode="eventlet")
@app.route('/')
@socketio.on('connect')
def handle_connect():
  #user connect place 
  print("a user has connected")
@socketio.on('disconnect')
def handle_disconnect():
    #user disconnected
    print("a user has disconnected")
@socketio.on("message")
def message(msg):
    print(msg)
@socketio.on("new_value")
def newval(dat):
    print(dat)
    socketio.emit("new_value", {"value":dat})
    tempstore['num']=dat

@socketio.on("getres")
def getres():
   storeddata=tempstore.get('num',None)
   socketio.emit("resval",{"value":storeddata})
   print("From Pricchatmonth GET")

if __name__ == "__main__":
    socketio.run(app,host="0.0.0.0", port=8081)
