
from flask import Flask, Response
from flask_socketio import SocketIO
import os

app = Flask(__name__)

last_value=None
tempstore={}
socketio=SocketIO(app,cors_allowed_origins="*",async_mode="gevent")
@app.route('/')
def index():
   print("Socket server is live")
   return "socket io say hello "
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


@socketio.on("serverurl")
def newserv(dat):
    print(dat)
    tempstore['urld']=dat


@socketio.on("getres")
def getres(mess):
    print(mess)
    storeddata=tempstore.get('num',None)
    socketio.emit("resval",{"value":storeddata})
    print("From Pricchatmonth GET") 
   
@socketio.on("geturl")
def geturl(mess):
    print(mess)
    storeddata=tempstore.get('urld',None)
    socketio.emit("nurl",{"url":storeddata})
    socketio.emit("captureframe","capturecapture")
   


if __name__ == "__main__":
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    port=int(os.environ.get("PORT",8081))
    server = pywsgi.WSGIServer(("0.0.0.0", port), app, handler_class=WebSocketHandler)
    server.serve_forever()
    print("Listening in port 8081 ,is online")
