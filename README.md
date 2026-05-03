# cs280_final_proj
Tian，确认几个最终细节：

【必须】
1. 物体bbox最长边=1，原点；相机距离1.5
2. matte材质用Principled BSDF: gray(0.5), roughness=1, IOR=1, 
   transmission=0（不要用Diffuse BSDF）
3. mask用0/255二值PNG（不是0/1）；depth用EXR米单位
4. 同scene所有view共享Cycles random seed
5. cond elevation=10°（不要0°，避免pose退化）

【今晚至少给1个scene救命；5/3补到3个物体】
优先: wine glass / bottle / Suzanne 各一个。ClearPose物体优先。

【明确不用做】
- 不做G1/G2 baseline渲染
- 不需要预算fundamental.json（我自己算）
- 不需要optical flow或多HDRI

有问题立刻call我，今晚6点的1个scene是死线。
