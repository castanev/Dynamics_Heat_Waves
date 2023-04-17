; Dry Core GCM
begin
  name = "exp1_Held_Suarez"
  uname = "u300.atmos_daily.nc"
  vname = "v300.atmos_daily.nc"
  ufile = addfile("/scratch/brown/castanev/DryCore_Wu/output/" + name + "/post_processed/output/" + uname, "r")
  vfile = addfile("/scratch/brown/castanev/DryCore_Wu/output/" + name + "/post_processed/output/" + vname, "r")

  u = ufile->u
  v = vfile->v

  printVarSummary(u)
  print(dimsizes(u))
  printVarSummary(v)
  print(dimsizes(v))

  uvmsg = 1e+36

  sf = new(dimsizes(u),float, uvmsg)
  vp = new(dimsizes(u),float, uvmsg)

  uv2sfvpf (u,v, sf,vp)
  ; printMinMax(sf,True)
  ; printMinMax(vp,True)

  output_name = "streamfunction_daily.nc"
  oh = addfile("/scratch/brown/castanev/DryCore_Wu/output/" + name + "/post_processed/output/" + output_name, "c")
  oh->SF = sf
  oh->VP = vp
end
exit

; ERA5
begin
  uname = "u_300_ERA5.nc"
  vname = "v_300_ERA5.nc"
      
  ufile = addfile("/scratch/brown/castanev/ERA5/" + uname, "r")
  vfile = addfile("/scratch/brown/castanev/ERA5/" + vname, "r")

  u = ufile->u_300
  v = vfile->v_300

  printVarSummary(u)
  print(dimsizes(u))
  printVarSummary(v)
  print(dimsizes(v))

  uvmsg = 1e+36

  sf = new(dimsizes(u),float, uvmsg)
  vp = new(dimsizes(u),float, uvmsg)

  uv2sfvpf (u,v, sf,vp)
  ; printMinMax(sf,True)
  ; printMinMax(vp,True)

  oh = addfile("ERA5/Data/" + "sf_vp_300_1948-2022.nc", "c")
  oh->SF = sf
  oh->VP = vp
end
exit



; CAM
begin
  uname = "u_300_CAM.nc"
  vname = "v_300_CAM.nc"
      
  ufile = addfile("CAM/Data/" + uname, "r")
  vfile = addfile("CAM/Data/" + vname, "r")

  u = ufile->u_300
  v = vfile->v_300


  printVarSummary(u)
  print(dimsizes(u))
  printVarSummary(v)
  print(dimsizes(v))

  uvmsg = 1e+36

  sf = new(dimsizes(u),float, uvmsg)
  vp = new(dimsizes(u),float, uvmsg)

  uv2sfvpf (u,v, sf,vp)
  ; printMinMax(sf,True)
  ; printMinMax(vp,True)

  oh = addfile("CAM/Data/" + "sf_vp_300.nc", "c")
  oh->SF = sf
  oh->VP = vp
end
exit


; NCEP
begin
  uname = "u_300_NCEP.nc"
  vname = "v_300_NCEP.nc"
      
  ufile = addfile("NCEP/Data/" + uname, "r")
  vfile = addfile("NCEP/Data/" + vname, "r")

  u = ufile->u_300
  v = vfile->v_300

  printVarSummary(u)
  print(dimsizes(u))
  printVarSummary(v)
  print(dimsizes(v))

  uvmsg = 1e+36

  sf = new(dimsizes(u),float, uvmsg)
  vp = new(dimsizes(u),float, uvmsg)

  uv2sfvpf (u,v, sf,vp)
  ; printMinMax(sf,True)
  ; printMinMax(vp,True)

  oh = addfile("NCEP/Data/" + "sf_vp_300_1948-2022.nc", "c")
  oh->SF = sf
  oh->VP = vp
end
exit



