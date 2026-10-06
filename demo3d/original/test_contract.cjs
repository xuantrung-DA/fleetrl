const fs=require('fs'),vm=require('vm'),assert=require('assert');process.chdir(__dirname);let html=fs.readFileSync('fleetrl_viewer.html','utf8');let s=html.split('<script>')[1].split('</script>')[0].replace('try{init()}catch(e){error(e)}','');const ctx={window:{},document:{},console};vm.createContext(ctx);vm.runInContext(s,ctx);let a=ctx.window.FleetRL;for(const name of ['A','B'])for(const n of [10,15,20]){let d=a.mockData(name,n),prev;for(const f of d.frames){a.validateFrame(f,d.map,prev);prev=f} console.log(name,n,d.frames.length,'frames pass');}
let d=a.mockData('A',15);let bad=structuredClone(d.frames[0]);bad.robots[2].cell=bad.robots[0].cell;assert.throws(()=>a.validateFrame(bad,d.map));bad=structuredClone(d.frames[0]);bad.robots[0].cell=[8,6];assert.throws(()=>a.validateFrame(bad,d.map));bad=structuredClone(d.frames[0]);bad.tasks[0].created=100;assert.throws(()=>a.validateFrame(bad,d.map));console.log('Duplicate occupancy, wall, future task rejected');

// Regression: raw JSONL/list must use the loaded map, not Array.prototype.map.
ctx.document.getElementById=()=>({});
vm.runInContext('buildWorld=()=>{};update=()=>{};map=presetMap("A");',ctx);
a.install(d.frames);
console.log('Raw frame array / JSONL map selection pass');
