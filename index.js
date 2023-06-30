fileButton.addEventListener('change',function(e){
    for (let i=0;i<e.target.files.length;i++){
        let imageFile = e.target.files[i];

        let storageRef = firebase.storage().ref("Images/"+imageFile.name);
        let task = storageRef.put(imageFile)
        task.on('state_changed',function progress(snapshot){
            let percentage = snapshot.bytesTransferred/snapshot * 100;
            console.log(percentage);
            switch (snapshot.state){
                case firebase.storage.TaskState.PAUSED :
                    console.log("PAUSED");
                    break;
                case firebase.storage.TaskState.RUNNING :
                    console.log("RUNNING");
                    break;

                
            }

        })
    }
})