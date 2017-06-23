var express = require('express');
var expressMongoRest = require('express-mongo-rest');
var app = express();

app.use('/api', expressMongoRest('mongodb://hero:hackathon2017@localhost:20984/leaderboards'));

/* serves main page */
app.get("/", function(req, res) {
    res.sendFile('index.html', {root: __dirname});
});

/* serves all the static files */
app.get(/^(.+)$/, function(req, res){ 
    console.log('static file request : ' + req.params);
    res.sendfile( __dirname + req.params[0]); 
});

var server = app.listen(64304, function () {
    console.log('Listening on Port', server.address().port)
})