const express = require("express")
const bodyParser = require("body-parser");

const app = express();

app.use(bodyParser.json())

const port = 3000;

const multer = require("multer");

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'images')
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname)
    }
})

const upload = multer({ storage: storage })

const {spawn} = require("child_process")




const gunDetectScript = (fileName) =>{
    return new Promise( resolve => {
        const gunDetectService = spawn("python3",["src/YoloGunDetect/main.py",`${fileName}`])

        gunDetectService.stdout.on("data",(data)=>{
            if (String(data).includes("Gun detected.")){
                resolve(true)
            }
            if (String(data).includes("Gun not detected")){
                resolve(false)
            }
        })
    })


}

app.post('/detectGun', upload.single("image"),async (req, res) => {

    const response = await gunDetectScript(req.file.originalname)


    res.send({gunDetected: response})
})

app.listen(port);
console.log("server is up ")