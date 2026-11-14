"""Human-specified reusable subtask descriptions, instantiated with scene entities."""
TEMPLATES={
 0:'Reach for and grasp {subject} from {target}.',
 1:'Carry and place {subject} at {target}; includes transport and release.',
 2:'Reach for the handle or control of {subject}.',
 3:'Open {subject}.',
 4:'Close {subject}.',
 5:'Slide or reposition the rack/drawer of {subject}.',
 6:'Turn on or start {subject}.',
 7:'Turn off or stop {subject}.',
 8:'Adjust the setting, temperature or spout of {subject}.',
 9:'Navigate toward {subject}.',
 10:'Pour or dump the contents of {subject} into {target}.',
 11:'Use {subject} to stir or mix {target}.',
 12:'Use or move {subject} to wash, rinse or scrub {target}.',
 13:'Use {subject} to press {target}.',
 14:'Tilt, rotate or shake {subject} with its contents.',
 15:'Align or stack {subject} relative to {target}.',
 16:'Cut or slice {target} using {subject}.',
 17:'Wait while {subject} operates.'}

PRIORITY_TASKS=['AddIceCubes','PickPlaceCounterToCabinet','OpenDrawer','PickPlaceCabinetToCounter',
 'CloseMicrowave','SlideDishwasherRack','OpenBlenderLid','StartCoffeeMachine','TurnOnStove',
 'AdjustWaterTemperature','NavigateKitchen','ScrubBowl','StirVegetables','PressChicken','TiltPan',
 'DumpLeftovers','FillKettle','RinseBowls','StackCans','StackBowlsCabinet','ReheatMeal',
 'OvenBroilFish','PlaceEqualIceCubes','TurnSinkSpout','TurnOffSinkFaucet',
 'PickPlaceFridgeShelfToDrawer','LoadDishwasher','ScalePortioning','RotatePan','PrepareSmoothie']

def allowed_templates(task):
    if task in ['OpenBlenderLid','CloseBlenderLid','OpenElectricKettleLid','CloseElectricKettleLid']:
        return [0,1,2,3,4]
    if task=='NavigateKitchen':return [9]
    if task.startswith('PickPlace') or task in ['CheesyBread','CoffeeServeMug','CoffeeSetupMug','PackDessert']:
        return [0,1]
    if task.startswith('Open'):return [2,3]
    if task.startswith('Close'):return [2,4]
    if task.startswith('Slide'):return [2,5]
    if task.startswith('TurnOn') or task=='StartCoffeeMachine':return [2,6]
    if task.startswith('TurnOff') and task!='TurnOffSimmeredSauceHeat':return [2,7]
    if task.startswith('Adjust') or task in ['LowerHeat','PreheatOven','TurnSinkSpout']:
        return [2,6,7,8]
    if task in ['AddIceCubes','PlaceEqualIceCubes','PlaceIceInCup','AddSugarCubes']:
        return [0,1]
    return list(TEMPLATES)

TASK_GUIDANCE={
 'AddIceCubes':'Transfer each actual ice cube once from bowl to blender. Determine identity/order from its observed motion, not its number. Finish the current transfer before returning for the next cube; do not invent extra transfers during pauses.',
 'PlaceEqualIceCubes':'Distribute the four distinct cubes into the two glasses as instructed. Keep each cube identity through its transfer and use the actual destination glass. A cube still in the bowl cannot simultaneously be transported.',
 'StirVegetables':'First pick and place the two vegetables into the pot, then retrieve the spatula and stir them. Do not label the early vegetable transfers as taking the spatula.',
 'DumpLeftovers':'Distinguish the contents destination from the empty-container destination. Follow the instruction: dump the contents onto the counter, then carry the emptied container to the sink.',
 'RinseBowls':'Handle the bowls in their observed order: take a bowl to the running water, rinse it there, and return/place it as shown, then handle the next bowl. A bowl still on the counter is not being rinsed. Faucet operations are separate phases.',
 'StackCans':'Preserve can identities and stack onto the actually supporting can. Do not swap the moved can with the stationary supporting can; the moving-entity cues distinguish them.',
 'StackBowlsCabinet':'If the smaller bowl is first stacked into the larger bowl on the counter, label that stacking before carrying the combined bowls into the cabinet. Do not call the early counter-side stacking a placement inside the cabinet.',
 'ReheatMeal':'Retrieve the meal from the fridge, transport it to the microwave, then close the microwave and start heating as instructed. The fridge is the source, not the destination of the microwave transfer. Do not insert closing while the meal is still being carried.',
 'OpenBlenderLid':'The lid is detachable: grasp/remove the lid from the blender, then carry and place that lid on the nearby counter. Do not label its entire transport as merely opening the blender.',
 'TurnOnStove':'Approach the correct knob, then turn it to start the burner. Once recorded stove-joint motion begins, include the turn-on operation instead of labelling the whole clip as reaching.',
 'TiltPan':'Grasp the pan handle, tilt the pan to move its contents, then return/place the pan on the stove. The middle phase is tilting, not continued reaching.',
 'PressChicken':'Retrieve the spatula, press it against the chicken for a while, then release/return it according to the actual demonstration. Include both tool and chicken during pressing.',
 'NavigateKitchen':'This task is only navigation. The destination fixture is the relevant entity and belongs in the subject slot for the navigation template.'}
