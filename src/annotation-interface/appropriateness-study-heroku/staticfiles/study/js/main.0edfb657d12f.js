
$(document).ready(function () {
    render_posts()
    $('#annotated-form').submit(function (e) {
        $(".error").remove();
        // var is_explanation_valid= validate_explanation()
        // if(!is_explanation_valid) {
        //     e.preventDefault();
        //    $('#explanationtextarea').before('<div class="error alert-danger">Please enter an explanation for your answer (at least 50 characters) </div>');
        //    return false;
        // } else {
        //     return true;
        // }
    });

});


function render_posts() {
    var post = $('#hidden_post').val();
    console.log(post);
    $('#selftext').html(post['selftext']);
}


// function validate_explanation() {
//       synergy =$('input[type=radio][name=synergyQuestion]:checked').val();
//       if (synergy !== '1' &&   !$.trim($("#explanationtextarea").val()) ) {
//            return false
//       }
//        return true
//    }
