#pragma once
#include "CommonInclude.h"
#include "wiInput.h"

namespace wi::input::xinput
{
	// GGMAX 2.15: how often an already-known-EMPTY XInput slot is re-probed, in frames
	// (staggered, at most one empty slot per frame). Connected pads are always polled every
	// frame. 0 = stock every-frame polling of all four slots. See wiXInput.cpp for why.
	extern uint32_t gg_xinput_rescan_frames;

	// Call once per frame to read and update controller states
	void Update();

	// Returns how many gamepads can Xinput handle
	int GetMaxControllerCount();

	// Returns whether the controller identified by index parameter is available or not.
	//	Id state parameter is not nullptr, and the controller is available, the state will be written into it
	bool GetControllerState(wi::input::ControllerState* state, int index);

	// Sends feedback data for the controller identified by index parameter to output
	void SetControllerFeedback(const wi::input::ControllerFeedback& data, int index);
}
