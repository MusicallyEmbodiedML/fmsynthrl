#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/synth/FMSynth.hpp"
#include "src/memlp/MLP.h"
#include "src/memlp/ReplayMemory.hpp"
#include "src/memlp/OrnsteinUhlenbeckNoise.h"

#include <memory>

volatile bool core1_ready = false;

//setup networks

constexpr size_t bias=1;

static const std::vector<ACTIVATION_FUNCTIONS> layers_activfuncs = {
    RELU, RELU, TANH
};

const size_t stateSize = 3;
const size_t actionSize = kN_synthparams;

const std::vector<size_t> actor_layers_nodes = {
    stateSize + bias,
    10, 10,
    actionSize
};

const std::vector<size_t> critic_layers_nodes = {
    stateSize + actionSize + bias,
    10, 10,
    1
};

const bool use_constant_weight_init = false;
const float constant_weight_init = 0;

std::shared_ptr<MLP<float> > actor, actorTarget, critic, criticTarget;

const float discountFactor = 0.95;
const float learningRate = 0.005;
const float smoothingAlpha = 0.005;

struct trainRLItem {
    std::vector<float> state ;
    std::vector<float> action;
    float reward;
    std::vector<float> nextState;
};

std::unique_ptr<FMSynth> pFMSynth;
std::vector<float> action;

ReplayMemory<trainRLItem> replayMem;

std::vector<float> actorOutput, criticOutput;
std::vector<float> criticInput(critic_layers_nodes[0]);

std::vector<float> criticLossLog, actorLossLog, log1;



//controls
float gain=1.f;
std::vector<float> joystick = {0,0,0};
bool rpSwitchState = 0;
bool rpSwitchState2 = 0;
size_t switchTs=0;
size_t switch2Ts=0;



void optimise() {
    std::vector<trainRLItem> sample = replayMem.sample(4);

    //run sample through critic target, build training set for critic net
    MLP<float>::training_pair_t ts;
    for(size_t i = 0; i < sample.size(); i++) {
        //---calculate y
        //--calc next-state-action pair
        //get next action from actorTarget given next state
        auto nextStateInput =  sample[i].nextState;
        nextStateInput.push_back(1.f); // bias
        actorTarget->GetOutput(nextStateInput, &actorOutput);
        //use criticTarget to estimate value of next action given next state
        for(size_t j=0; j < stateSize; j++) {
          criticInput[j] = sample[i].nextState[j];
        }
        for(size_t j=0; j < actionSize; j++) {
          criticInput[j+stateSize] = actorOutput[j];
        }
        criticInput[criticInput.size()-1] = 1.f; //bias
        // criticInput[0] = sample[i].nextState[0];
        // criticInput[1] = actorOutput[0];
        // criticInput[2] = 1.f; //bias
        criticTarget->GetOutput(criticInput, &criticOutput);

        //calculate expected reward
        float y = sample[i].reward + (discountFactor *  criticOutput[0]);
        // std::cout << "[" << i << "]: y: " << y << std::endl;

        //use criticTarget to estimate value of next action given next state
        for(size_t j=0; j < stateSize; j++) {
          criticInput[j] = sample[i].state[j];
        }
        for(size_t j=0; j < actionSize; j++) {
          criticInput[j+stateSize] = sample[i].action[j];
        }
        criticInput[criticInput.size()-1] = 1.f; //bias
        // criticInput[0] = sample[i].state[0];
        // criticInput[1] = sample[i].action[0];
        // criticInput[2] = 1.f; //bias
        ts.first.push_back(criticInput);
        ts.second.push_back({y});
    }
    float loss = critic->Train(ts, learningRate, 1);
    criticLossLog.push_back(loss);

    //update the actor

    //for each memory in replay memory sample, and get grads from critic
    std::vector<float> actorLoss(actionSize, 0.f);
    std::vector<float> gradientLoss= {1.f};
    
    for(size_t i = 0; i < sample.size(); i++) {
        //numerical diff
        constexpr float eps = 1e-4;
        //use criticTarget to estimate value of next action given next state
        for(size_t j=0; j < stateSize; j++) {
          criticInput[j] = sample[i].nextState[j];
        }
        for(size_t j=0; j < actionSize; j++) {
          criticInput[j+stateSize] = sample[i].action[j];
        }
        criticInput[criticInput.size()-1] = 1.f; //bias

        critic->CalcGradients(criticInput, gradientLoss);
        std::vector<float> l0Grads = critic->m_layers[0].GetGrads();

        for(size_t j=0; j < actionSize; j++) {
          actorLoss[j] = l0Grads[j+stateSize];
        }

        // // criticInput[0] = sample[i].nextState[0];
        // // criticInput[1] = sample[i].action[0];
        // // criticInput[2] = 1.f; //bias
        // critic->GetOutput(criticInput, &criticOutput);
        // float diff1 = criticOutput[0];

        // for(size_t j=0; j < criticInput.size(); j++) {
        //   criticInput[j] += eps;
        // }
        // // criticInput[0] += eps;
        // // criticInput[1] += eps;
        // // criticInput[2] += eps;
        // critic->GetOutput(criticInput, &criticOutput);
        // float diff2 = criticOutput[0];
        // actorLoss  += ((diff2-diff1)/eps);
    }
    float totalLoss = 0.f;
    for(size_t j=0; j < actorLoss.size(); j++) {
      actorLoss[j] /= sample.size();
      actorLoss[j] = -actorLoss[j];
      totalLoss += actorLoss[j];
    }
    // actorLossLog.push_back(actorLoss);
    // actorLoss = -actorLoss;
    Serial.printf("Actor loss: %f\n", totalLoss);


    //back propagate the actor loss
    for(size_t i = 0; i < sample.size(); i++) {
        auto actorInput = sample[i].state;
        actorInput.push_back(bias); 

        actor->ApplyLoss(actorInput, actorLoss, learningRate);
    }
    // std::cout << "Updated actor\n";

    //soft update the target networks
    criticTarget->SmoothUpdateWeights(critic, smoothingAlpha);
    actorTarget->SmoothUpdateWeights(actor, smoothingAlpha);

}

void replayMemSwitch() {
  int switchState = digitalRead(24);
  auto now = millis();
  if (switchState != rpSwitchState && (now-switchTs > 100)) {
    rpSwitchState = switchState;
    switchTs=now;
    Serial.printf("rp switch %d\n", rpSwitchState);
    if (switchState == 0) {
      //add to replay mem
        auto currentState = joystick;
        
        trainRLItem trainItem = {currentState, action, 1.f, currentState};
        replayMem.add(trainItem, millis());

        Serial.println("Loved that mapping!");

    }
  }

}

void replayMemSwitch2() {
  int switchState2 = digitalRead(25);
  auto now = millis();
  Serial.printf("switch 2: %d\n", switchState2);
  if (switchState2 != rpSwitchState2 && (now-switch2Ts > 100)) {
    rpSwitchState2 = switchState2;
    switch2Ts=now;
    Serial.printf("rp switch 2: %d\n", rpSwitchState2);
    if (switchState2 == 0) {
      //add to replay mem
        auto currentState = joystick;
        
        trainRLItem trainItem = {currentState, action, -1.f, currentState};
        replayMem.add(trainItem, millis());

        Serial.println("That mapping sucks!");


    }
  }

}

void setup()
{
    Serial.begin(115200);
    Serial.println("Core 0: Waiting for Core 1 setup...");
    while (!core1_ready) {
        delay(1);
    }
    Serial.println("Core 0: Starting main loop");

    //init networks
    actor = std::make_shared<MLP<float> > (
        actor_layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    actorTarget = std::make_shared<MLP<float> > (
        actor_layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    critic = std::make_shared<MLP<float> > (
        critic_layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );
    criticTarget = std::make_shared<MLP<float> > (
        critic_layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );


    pinMode(24, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(24), replayMemSwitch,
                    CHANGE);
    pinMode(25, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(25), replayMemSwitch2,
                    CHANGE);

}

size_t trainDiv = 0;

double theta = 0.15;  // Reversion speed
double mu = 0.0;      // Long-term mean
double sigma = 0.3;   // Noise intensity
double dt = 0.01;     // Time step

OrnsteinUhlenbeckNoise ou_noise(theta, mu, sigma, dt);

float knobL, knobR;

void loop()
{
  gain = analogRead(47)/1024.f;
  joystick[0] = analogRead(40)/1024.f;
  joystick[1] = analogRead(41)/1024.f;
  joystick[2] = analogRead(42)/1024.f;
  knobL = analogRead(45)/1024.f;
  knobR = analogRead(46)/1024.f;
  std::vector<float> currentState = joystick;
  currentState.push_back(bias);
  std::vector<float> actorOutput;
  actorTarget->GetOutput(currentState, &actorOutput);
  for(size_t i=0; i < actorOutput.size(); i++) {
        const float noise = ou_noise.sample() * knobL;
        actorOutput[i] += noise;
  }
  action = actorOutput;
  pFMSynth->mapParameters(actorOutput);
  pFMSynth->UpdateParams();
  if (trainDiv++ >= static_cast<size_t>(1 + (knobR * 19.f)))
  {
    trainDiv=0;
    optimise();
  }
  delay(50);
}


//core 1 variables

size_t dspcount=0;
stereosample_t dsploop(stereosample_t x) {
    x.L = pFMSynth->process() * gain;
    // x.L = rand() / (float)RAND_MAX;
    x.R = x.L;
    if (20000 == dspcount++) {
      dspcount=0;
      Serial.println(x.L);
    }
    return x;
}

void setup1()
{
    while(!Serial) {}
    Serial.println("Core 1: Starting setup");

    pFMSynth = std::make_unique<FMSynth>(kSampleRate);

    AudioDriver_Output::Setup();
    AudioDriver_Output::SetCallback(dsploop);
    core1_ready = true;
    Serial.println("Core 1: Setup complete");
}

void loop1()
{

}

extern "C" int getentropy (void * buffer, size_t how_many) {
    uint8_t* pBuf = (uint8_t*) buffer;
    while(how_many--) {
        uint8_t rand_val = rp2040.hwrand32() % UINT8_MAX;
        *pBuf++ = rand_val;
    }
    return 0; // return "no error". Can also do EFAULT, EIO, ENOSYS
}

